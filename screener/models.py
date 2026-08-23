from django.db import models


class JobDescription(models.Model):
    STATUS_OPEN = 'open'
    STATUS_FILLED = 'filled'
    STATUS_CLOSED = 'closed'
    STATUS_CHOICES = [
        (STATUS_OPEN, 'Open'),
        (STATUS_FILLED, 'Filled'),
        (STATUS_CLOSED, 'Closed'),
    ]

    title = models.CharField(max_length=255)
    raw_text = models.TextField()

    # Structured output of screener.llm.extraction.extract_job_requirements:
    # {"required_skills": [...], "preferred_skills": [...],
    #  "min_experience_years": int|None, "education_requirement": str|None,
    #  "eligibility_criteria": [str, ...]}
    extracted_requirements = models.JSONField(default=dict, blank=True)

    # Open roles are the only ones offered as a cross-role recommendation
    # target (see views.check_other_roles / _build_shortlist_context) --
    # filled/closed roles stop absorbing new "may fit better elsewhere"
    # suggestions, but existing Match records and the shortlist itself are
    # untouched, so past results stay intact after a role closes.
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_OPEN)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Resume(models.Model):
    candidate_name = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='resumes/')
    raw_text = models.TextField(blank=True)

    # SHA-256 of raw_text, used to reject re-uploading the same resume as a
    # second Resume row (which would otherwise create duplicate, independently
    # scored entries in every shortlist). Hashing the extracted text rather
    # than the raw file bytes also catches a re-saved/re-exported copy of the
    # same content. Blank until text_extraction has run.
    content_hash = models.CharField(max_length=64, blank=True, db_index=True)

    # PII-stripped text (name/age/gender/address markers removed) — this is
    # what gets sent to the scoring chain, never raw_text. See
    # screener/llm/redaction.py.
    redacted_text = models.TextField(blank=True)

    # Structured output of screener.llm.extraction.extract_resume_profile:
    # list[str], list[{title, company, duration, summary}], list[{degree, institution, year}]
    extracted_skills = models.JSONField(default=list, blank=True)
    extracted_experience = models.JSONField(default=list, blank=True)
    extracted_education = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.candidate_name or self.file.name


class Match(models.Model):
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE, related_name='matches')
    job_description = models.ForeignKey(
        JobDescription, on_delete=models.CASCADE, related_name='matches'
    )

    # Per-dimension rubric scoring (1-10) — see screener/llm/scoring.py
    skills_subscore = models.FloatField(default=0)
    experience_subscore = models.FloatField(default=0)
    education_subscore = models.FloatField(default=0)

    skills_justification = models.TextField(blank=True)
    experience_justification = models.TextField(blank=True)
    education_justification = models.TextField(blank=True)

    # {"skills": [...], "experience": [...], "education": [...]} — quoted
    # resume snippets the LLM cited as evidence for each dimension.
    evidence_quotes = models.JSONField(default=dict, blank=True)

    # [{"skill": str, "evidence_level": "Strong"|"Partial"|"Weak"|"None",
    #   "evidence_quote": str|None}, ...] — one entry per required skill,
    # graded individually (distinct from the aggregate skills_subscore
    # above). Drives the 4-state skill matrix on the dashboard.
    skill_evidence = models.JSONField(default=list, blank=True)

    # Hybrid score components (0-10) and final blended score.
    deterministic_score = models.FloatField(default=0)
    llm_semantic_score = models.FloatField(default=0)
    final_score = models.FloatField(default=0)

    # [{"question": str, "targets": str}] generated from low-scoring dimensions.
    interview_questions = models.JSONField(default=list, blank=True)

    # Hard pass/fail gate (min CGPA/percentage, graduation status, mandatory
    # experience cutoffs) — see JobDescription.extracted_requirements'
    # eligibility_criteria and screener/llm/scoring.py. Ineligible candidates
    # are excluded from the ranked shortlist in the dashboard.
    is_eligible = models.BooleanField(default=True)
    # [{"criterion": str, "met": bool, "evidence": str}, ...]
    eligibility_details = models.JSONField(default=list, blank=True)

    # True once a candidate has withdrawn (or been marked withdrawn by a
    # recruiter) from consideration for this specific role. Kept as a
    # record rather than deleted, so the audit trail and PDF report still
    # reflect that they applied -- just excluded from the active shortlist,
    # eligibility split, interview-target pool math, and skill matrix.
    withdrawn = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('resume', 'job_description')
        ordering = ['-final_score']

    def __str__(self):
        return f'{self.resume} x {self.job_description} = {self.final_score}'
