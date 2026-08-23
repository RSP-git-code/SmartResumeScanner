"""Orchestrates the full extract -> redact -> index -> score -> generate
flow and persists results onto the Django models. This is the single entry
point views (and scripts) should call -- nothing outside this module should
need to import the individual chain modules directly.

Runs synchronously: fine at take-home/demo scale (a handful of resumes per
match run). A production version would move `run_match` onto a task queue
(Celery) so uploads don't block on multiple LLM round-trips.
"""

from . import extraction, interview, redaction, scoring
from .rag import build_resume_retriever


def process_resume(resume):
    """Structured extraction + redaction for a newly uploaded resume.
    Idempotent -- safe to call again if extraction needs to be re-run.
    """
    profile = extraction.extract_resume_profile(resume.raw_text)
    resume.candidate_name = profile.candidate_name or resume.candidate_name
    resume.extracted_skills = profile.skills
    resume.extracted_experience = [e.model_dump() for e in profile.experience]
    resume.extracted_education = [e.model_dump() for e in profile.education]
    resume.redacted_text = redaction.redact_resume_text(resume.raw_text)
    resume.save()
    return resume


def process_job_description(job_description):
    requirements = extraction.extract_job_requirements(job_description.raw_text)
    job_description.extracted_requirements = requirements.model_dump()
    job_description.save()
    return job_description


def run_match(resume, job_description):
    """Runs RAG-grounded multi-dimensional scoring + hybrid combine +
    interview-question generation for one resume/JD pair, and upserts the
    Match row.
    """
    from screener.models import Match  # local import avoids a hard app-registry dependency at module load

    if not resume.redacted_text:
        process_resume(resume)
    if not job_description.extracted_requirements:
        process_job_description(job_description)

    requirements = job_description.extracted_requirements
    required_skills = requirements.get("required_skills", [])
    preferred_skills = requirements.get("preferred_skills", [])
    retriever = build_resume_retriever(resume.id, job_description.id, resume.redacted_text)

    scores = scoring.score_all_dimensions(
        retriever,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_experience_years=requirements.get("min_experience_years"),
        education_requirement=requirements.get("education_requirement"),
    )
    skills_result = scores.skills
    experience_result = scores.experience
    education_result = scores.education

    llm_semantic_score = round(
        (skills_result.score + experience_result.score + education_result.score) / 3, 2
    )
    deterministic_score = scoring.deterministic_skill_overlap(
        resume.extracted_skills,
        required_skills,
        preferred_skills,
    )
    final_score = scoring.combine_scores(llm_semantic_score, deterministic_score)

    interview_questions = interview.generate_interview_questions(
        job_title=job_description.title,
        skills_score=skills_result.score,
        skills_justification=skills_result.justification,
        experience_score=experience_result.score,
        experience_justification=experience_result.justification,
        education_score=education_result.score,
        education_justification=education_result.justification,
    )

    match, _ = Match.objects.update_or_create(
        resume=resume,
        job_description=job_description,
        defaults=dict(
            skills_subscore=skills_result.score,
            experience_subscore=experience_result.score,
            education_subscore=education_result.score,
            skills_justification=skills_result.justification,
            experience_justification=experience_result.justification,
            education_justification=education_result.justification,
            evidence_quotes={
                "skills": skills_result.evidence_quotes,
                "experience": experience_result.evidence_quotes,
                "education": education_result.evidence_quotes,
            },
            deterministic_score=deterministic_score,
            llm_semantic_score=llm_semantic_score,
            final_score=final_score,
            interview_questions=interview_questions,
        ),
    )
    return match
