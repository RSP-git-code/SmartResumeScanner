"""Django template views for the dashboard. Thin wrappers around
text_extraction.py and llm/pipeline.py, same as api.py -- see that module's
docstring.
"""

import hashlib
import logging

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from . import text_extraction
from .llm import pipeline
from .models import JobDescription, Resume

logger = logging.getLogger(__name__)


def _score_badge_class(score):
    """Color-codes a 0-10 score for at-a-glance scanning of the shortlist."""
    if score >= 8:
        return 'bg-success'
    if score >= 5:
        return 'bg-warning text-dark'
    return 'bg-danger'


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


def job_detail(request, job_description_id):
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    matches = list(jd.matches.select_related('resume').order_by('-final_score'))
    other_job_count = JobDescription.objects.exclude(id=jd.id).count()

    for match in matches:
        match.skills_pct = match.skills_subscore * 10
        match.experience_pct = match.experience_subscore * 10
        match.education_pct = match.education_subscore * 10
        match.score_badge_class = _score_badge_class(match.final_score)

        other_role_matches = list(
            match.resume.matches.exclude(job_description=jd)
            .select_related('job_description')
            .order_by('-final_score')
        )
        for om in other_role_matches:
            om.score_badge_class = _score_badge_class(om.final_score)
        match.other_role_matches = other_role_matches
        match.better_fit_elsewhere = (
            other_role_matches[0]
            if other_role_matches and other_role_matches[0].final_score > match.final_score
            else None
        )

    matched_resume_ids = {m.resume_id for m in matches}
    unmatched_resumes = Resume.objects.exclude(id__in=matched_resume_ids)

    eligible_matches = [m for m in matches if m.is_eligible]
    ineligible_matches = [m for m in matches if not m.is_eligible]

    required_skills = (jd.extracted_requirements or {}).get('required_skills', [])
    skill_matrix = [
        {
            'skill': entry.get('skill', ''),
            'weight': entry.get('weight'),
            'has_skill': [
                entry.get('skill', '').lower() in {s.lower() for s in m.resume.extracted_skills}
                for m in eligible_matches
            ],
        }
        for entry in required_skills
    ]

    return render(request, 'screener/job_detail.html', {
        'jd': jd,
        'matches': eligible_matches,
        'ineligible_matches': ineligible_matches,
        'unmatched_resumes': unmatched_resumes,
        'skill_matrix': skill_matrix,
        'other_job_count': other_job_count,
    })


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
        other_jds = list(JobDescription.objects.exclude(id=jd.id))
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
