from django.db import models


class JobDescription(models.Model):
    title = models.CharField(max_length=255)
    raw_text = models.TextField()

    # Structured output of screener.llm.extraction.extract_job_requirements:
    # {"required_skills": [...], "preferred_skills": [...],
    #  "min_experience_years": int|None, "education_requirement": str|None}
    extracted_requirements = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class Resume(models.Model):
    candidate_name = models.CharField(max_length=255, blank=True)
    file = models.FileField(upload_to='resumes/')
    raw_text = models.TextField(blank=True)

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

    # Hybrid score components (0-10) and final blended score.
    deterministic_score = models.FloatField(default=0)
    llm_semantic_score = models.FloatField(default=0)
    final_score = models.FloatField(default=0)

    # [{"question": str, "targets": str}] generated from low-scoring dimensions.
    interview_questions = models.JSONField(default=list, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('resume', 'job_description')
        ordering = ['-final_score']

    def __str__(self):
        return f'{self.resume} x {self.job_description} = {self.final_score}'
