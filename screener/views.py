"""Django template views for the dashboard. Thin wrappers around
text_extraction.py and llm/pipeline.py, same as api.py -- see that module's
docstring.
"""

import hashlib
import logging

from django.contrib import messages
from django.db.models import Prefetch
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render

from . import reports, text_extraction
from .llm import pipeline
from .models import JobDescription, Match, Resume

logger = logging.getLogger(__name__)


def _score_badge_class(score):
    """Color-codes a 0-10 score for at-a-glance scanning of the shortlist."""
    if score >= 8:
        return 'bg-success'
    if score >= 5:
        return 'bg-warning text-dark'
    return 'bg-danger'


def _qualification_band(score):
    """Same 8/5 thresholds as _score_badge_class, as a recruiter-facing label."""
    if score >= 8:
        return 'Strong Match'
    if score >= 5:
        return 'Moderate Match'
    return 'Below Cutoff'


def _skill_evidence_level(match, skill_name):
    """Looks up how strongly `match` demonstrates `skill_name`, from the
    per-skill grading in Match.skill_evidence (Strong/Partial/Weak/None --
    see screener/llm/scoring.py). Matched by normalized name since the LLM
    echoes the skill name back rather than an id. Falls back to 'None' if
    the skill is missing from skill_evidence (e.g. rows scored before this
    field existed).
    """
    skill_name = skill_name.strip().lower()
    for entry in match.skill_evidence:
        if entry.get('skill', '').strip().lower() == skill_name:
            return entry.get('evidence_level', 'None')
    return 'None'


def _gap_summary(match):
    """One-line "why not a stronger match" explanation for anything short of
    a Strong Match, built entirely from data already computed during scoring
    (no extra LLM call) -- names the single weakest dimension and quotes its
    existing justification.
    """
    dimensions = [
        ('Skills', match.skills_subscore, match.skills_justification),
        ('Experience', match.experience_subscore, match.experience_justification),
        ('Education', match.education_subscore, match.education_justification),
    ]
    weakest_name, weakest_score, weakest_justification = min(dimensions, key=lambda d: d[1])
    justification = weakest_justification.strip()
    if len(justification) > 160:
        justification = justification[:157].rstrip() + '...'
    return f'Scored {match.final_score}/10. Main gap -- {weakest_name} ({weakest_score}/10): {justification}'


def home(request):
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        text = request.POST.get('text', '').strip()
        file_obj = request.FILES.get('file')

        if not title or not (text or file_obj):
            messages.error(request, 'Title and either pasted text or a file are required.')
            return redirect('home')

        try:
            raw_text = text or text_extraction.extract_text(file_obj, filename=file_obj.name)
        except ValueError as exc:
            messages.error(request, str(exc))
            return redirect('home')

        jd = JobDescription.objects.create(title=title, raw_text=raw_text)
        try:
            pipeline.process_job_description(jd)
        except Exception:
            logger.exception('Requirement extraction failed for JobDescription %s', jd.id)
            messages.warning(
                request,
                'Job description saved, but requirement extraction failed -- '
                'use "Retry Extraction" on its page to try again.',
            )
        return redirect('job-detail', job_description_id=jd.id)

    job_descriptions = JobDescription.objects.order_by('-created_at')
    return render(request, 'screener/home.html', {'job_descriptions': job_descriptions})


def resume_pool(request):
    """Every uploaded resume across all jobs, and which roles it's been
    matched against -- resumes are a shared pool matchable against any open
    JD (see check_other_roles), but that's easy to lose track of from any
    single job's page. Read-only: matching itself still happens from a
    job's own page (Upload Resumes / Run Matching / Check other roles).
    """
    resumes = Resume.objects.order_by('-created_at').prefetch_related(
        Prefetch('matches', queryset=Match.objects.select_related('job_description').order_by('-final_score'))
    )
    for r in resumes:
        for m in r.matches.all():
            m.score_badge_class = _score_badge_class(m.final_score)
            m.final_score_pct = round(m.final_score * 10)
    return render(request, 'screener/resume_pool.html', {'resumes': resumes})


def _build_shortlist_context(jd, request):
    """Everything the dashboard and the PDF export need: ranked matches with
    display-ready fields, eligibility split, qualification-band counts, and
    the interview-target/cutoff planner. Shared so the exported report can
    never drift from what's on screen.
    """
    matches = list(jd.matches.select_related('resume').order_by('-final_score'))
    other_job_count = JobDescription.objects.exclude(id=jd.id).filter(status=JobDescription.STATUS_OPEN).count()

    for match in matches:
        match.skills_pct = match.skills_subscore * 10
        match.experience_pct = match.experience_subscore * 10
        match.education_pct = match.education_subscore * 10
        match.final_score_pct = round(match.final_score * 10)
        match.score_badge_class = _score_badge_class(match.final_score)
        match.qualification_band = _qualification_band(match.final_score)
        if match.qualification_band != 'Strong Match':
            match.gap_summary = _gap_summary(match)

        # Only open roles are offered as a "may fit better elsewhere"
        # suggestion -- a filled/closed role isn't accepting candidates.
        other_role_matches = list(
            match.resume.matches.exclude(job_description=jd)
            .filter(job_description__status=JobDescription.STATUS_OPEN)
            .select_related('job_description')
            .order_by('-final_score')
        )
        for om in other_role_matches:
            om.score_badge_class = _score_badge_class(om.final_score)
            om.final_score_pct = round(om.final_score * 10)
        match.other_role_matches = other_role_matches
        match.better_fit_elsewhere = (
            other_role_matches[0]
            if other_role_matches and other_role_matches[0].final_score > match.final_score
            else None
        )

    matched_resume_ids = {m.resume_id for m in matches}
    unmatched_resumes = Resume.objects.exclude(id__in=matched_resume_ids)

    # Withdrawn candidates are kept out of the shortlist/eligibility split,
    # band counts, interview-target math, and skill matrix entirely -- but
    # never deleted, so the record (and the PDF report) still shows they
    # applied. See Match.withdrawn and views.withdraw_candidate.
    active_matches = [m for m in matches if not m.withdrawn]
    withdrawn_matches = [m for m in matches if m.withdrawn]

    eligible_matches = [m for m in active_matches if m.is_eligible]
    ineligible_matches = [m for m in active_matches if not m.is_eligible]

    band_counts = {'strong': 0, 'moderate': 0, 'below_cutoff': 0}
    band_keys = {'Strong Match': 'strong', 'Moderate Match': 'moderate', 'Below Cutoff': 'below_cutoff'}
    for m in eligible_matches:
        band_counts[band_keys[m.qualification_band]] += 1

    # Interview-target planner: "if I want to interview N people, what score
    # cutoff gets me there?" -- pure computation over the already-ranked
    # eligible_matches, no new scoring involved. Candidates are never hidden
    # by this, only marked as in/out of the selected pool (?interview_target=N).
    total_eligible = len(eligible_matches)
    interview_target_options = [
        {'target': n, 'cutoff_score': eligible_matches[n - 1].final_score}
        for n in (5, 10, 15, 20)
        if n < total_eligible
    ]
    if total_eligible:
        interview_target_options.append({'target': total_eligible, 'cutoff_score': eligible_matches[-1].final_score})

    selected_target = request.GET.get('interview_target')
    selected_cutoff = None
    if selected_target:
        try:
            selected_target = int(selected_target)
        except ValueError:
            selected_target = None
        else:
            match_by_target = next((o for o in interview_target_options if o['target'] == selected_target), None)
            if match_by_target:
                selected_cutoff = match_by_target['cutoff_score']

    selected_cutoff_pct = None
    if selected_cutoff is not None:
        selected_cutoff_pct = round(selected_cutoff * 10)
        for m in eligible_matches:
            m.in_interview_pool = m.final_score >= selected_cutoff

    required_skills = (jd.extracted_requirements or {}).get('required_skills', [])
    skill_matrix = [
        {
            'skill': entry.get('skill', ''),
            'weight': entry.get('weight'),
            'evidence': [_skill_evidence_level(m, entry.get('skill', '')) for m in eligible_matches],
        }
        for entry in required_skills
    ]

    return {
        'jd': jd,
        'matches': eligible_matches,
        'ineligible_matches': ineligible_matches,
        'withdrawn_matches': withdrawn_matches,
        'unmatched_resumes': unmatched_resumes,
        'skill_matrix': skill_matrix,
        'other_job_count': other_job_count,
        'band_counts': band_counts,
        'interview_target_options': interview_target_options,
        'selected_target': selected_target,
        'selected_cutoff': selected_cutoff,
        'selected_cutoff_pct': selected_cutoff_pct,
    }


def job_detail(request, job_description_id):
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    context = _build_shortlist_context(jd, request)
    return render(request, 'screener/job_detail.html', context)


def export_report(request, job_description_id):
    """Downloads the same shortlist shown on the dashboard as a hiring-manager
    PDF -- built from the identical `_build_shortlist_context`, so it always
    matches whatever's on screen, including the current ?interview_target=.
    """
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    context = _build_shortlist_context(jd, request)
    pdf_bytes = reports.build_hiring_manager_report_pdf(
        jd,
        context['matches'],
        context['ineligible_matches'],
        context['band_counts'],
        selected_target=context['selected_target'],
        selected_cutoff=context['selected_cutoff'],
        withdrawn_count=len(context['withdrawn_matches']),
        skill_matrix=context['skill_matrix'],
    )
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    filename = f'{jd.title}-shortlist-report.pdf'.replace(' ', '-')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def update_job_status(request, job_description_id):
    """Marks a role Open/Filled/Closed. Doesn't touch existing Match records
    or the shortlist -- only affects whether the role is still offered as a
    cross-role recommendation target (see check_other_roles and
    _build_shortlist_context's other_role_matches filter).
    """
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    if request.method == 'POST':
        status = request.POST.get('status')
        if status in dict(JobDescription.STATUS_CHOICES):
            jd.status = status
            jd.save(update_fields=['status'])
            messages.success(request, f'Marked "{jd.title}" as {jd.get_status_display()}.')
        else:
            messages.error(request, 'Invalid status.')
    return redirect('job-detail', job_description_id=jd.id)


def withdraw_candidate(request, job_description_id, resume_id):
    """Toggles a candidate's withdrawn flag for this one role. Kept as a
    record rather than deleted -- see Match.withdrawn.
    """
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    resume = get_object_or_404(Resume, pk=resume_id)
    if request.method == 'POST':
        match = get_object_or_404(jd.matches, resume=resume)
        match.withdrawn = not match.withdrawn
        match.save(update_fields=['withdrawn'])
        if match.withdrawn:
            messages.success(request, f'Marked {resume} as withdrawn from "{jd.title}".')
        else:
            messages.success(request, f'Restored {resume} to the active shortlist for "{jd.title}".')
    return redirect('job-detail', job_description_id=jd.id)


def upload_resumes(request, job_description_id):
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    if request.method == 'POST':
        files = request.FILES.getlist('resumes')
        if not files:
            messages.error(request, 'Select at least one resume file.')
        uploaded_count = 0
        for f in files:
            try:
                raw_text = text_extraction.extract_text(f, filename=f.name)
            except ValueError as exc:
                messages.error(request, f'{f.name}: {exc}')
                continue

            content_hash = hashlib.sha256(raw_text.encode('utf-8')).hexdigest()
            existing = Resume.objects.filter(content_hash=content_hash).first()
            if existing:
                messages.info(
                    request,
                    f'Skipped {f.name}: already uploaded as '
                    f'"{existing.candidate_name or existing.file.name}".',
                )
                continue

            resume = Resume.objects.create(file=f, raw_text=raw_text, content_hash=content_hash)
            try:
                pipeline.process_resume(resume)
            except Exception:
                logger.exception('Resume extraction failed for Resume %s (%s)', resume.id, f.name)
                messages.warning(request, f'{f.name} uploaded, but extraction failed: check server logs.')
            uploaded_count += 1
        if uploaded_count:
            messages.success(request, f'Uploaded {uploaded_count} resume(s).')
    return redirect('job-detail', job_description_id=jd.id)


def retry_jd_extraction(request, job_description_id):
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    if request.method == 'POST':
        try:
            pipeline.process_job_description(jd)
        except Exception:
            logger.exception('Retry requirement extraction failed for JobDescription %s', jd.id)
            messages.warning(request, 'Extraction failed again -- check server logs.')
        else:
            messages.success(request, 'Requirements extracted successfully.')
    return redirect('job-detail', job_description_id=jd.id)


def run_matching(request, job_description_id):
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    if request.method == 'POST':
        matched_resume_ids = set(jd.matches.values_list('resume_id', flat=True))
        resumes = list(Resume.objects.exclude(id__in=matched_resume_ids))
        if not resumes:
            messages.info(request, 'All uploaded resumes are already matched against this job.')
        for resume in resumes:
            try:
                pipeline.run_match(resume, jd)
            except Exception:
                logger.exception('Matching failed for Resume %s x JobDescription %s', resume.id, jd.id)
                messages.warning(request, f'Matching failed for {resume}: check server logs.')
        if resumes:
            messages.success(request, f'Matched {len(resumes)} resume(s).')
    return redirect('job-detail', job_description_id=jd.id)


def check_other_roles(request, job_description_id, resume_id):
    """Scores one candidate against every other open job description, so a
    strong-but-not-quite-right-here candidate (e.g. applied for DevOps but
    is really a data analyst) can be surfaced for other open roles instead
    of just being screened out. Explicit per-candidate action rather than
    automatic, since cost scales with the number of open JDs.
    """
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    resume = get_object_or_404(Resume, pk=resume_id)
    if request.method == 'POST':
        other_jds = list(JobDescription.objects.exclude(id=jd.id).filter(status=JobDescription.STATUS_OPEN))
        if not other_jds:
            messages.info(request, 'No other open roles to compare against yet.')
        for other_jd in other_jds:
            try:
                pipeline.run_match(resume, other_jd)
            except Exception:
                logger.exception('Cross-role check failed for Resume %s x JobDescription %s', resume.id, other_jd.id)
                messages.warning(request, f'Cross-role check failed for {other_jd.title}: check server logs.')
        if other_jds:
            messages.success(request, f'Checked fit against {len(other_jds)} other role(s).')
    return redirect('job-detail', job_description_id=jd.id)
