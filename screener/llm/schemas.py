"""Pydantic schemas the LLM chains are forced to return via
`with_structured_output`. Keeping every LLM response typed is what lets the
rest of the pipeline (and the dashboard) trust the output instead of
re-parsing free text.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ExperienceEntry(BaseModel):
    title: str = Field(description="Job title, e.g. 'Backend Engineer'")
    company: str = Field(description="Employer name")
    duration: str = Field(description="e.g. '2021-2025 (4 years)'")
    summary: str = Field(description="1-2 sentence summary of responsibilities/impact")


class EducationEntry(BaseModel):
    degree: str = Field(description="e.g. 'B.Tech in Computer Science'")
    institution: str
    year: Optional[str] = Field(default=None, description="Graduation year, if stated")


class ResumeProfile(BaseModel):
    """Structured extraction target for a resume."""

    candidate_name: Optional[str] = Field(
        default=None, description="The candidate's full name, if present in the text"
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Flat list of distinct technical/professional skills mentioned",
    )
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[EducationEntry] = Field(default_factory=list)


class WeightedSkill(BaseModel):
    skill: str = Field(description="Skill name, normalized to common industry terminology")
    weight: int = Field(
        ge=1, le=5,
        description=(
            "Importance to the role, 1-5: 5 = explicitly must-have/critical, "
            "3 = clearly expected but not emphasized as critical, "
            "1 = mentioned only in passing"
        ),
    )


class JobRequirements(BaseModel):
    """Structured extraction target for a job description."""

    required_skills: list[WeightedSkill] = Field(
        default_factory=list,
        description="Explicitly required skills, each with an importance weight",
    )
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: Optional[int] = Field(
        default=None, description="Minimum years of experience required, if stated"
    )
    education_requirement: Optional[str] = Field(
        default=None, description="e.g. \"Bachelor's in Computer Science\""
    )


class DimensionScore(BaseModel):
    """Rubric-constrained sub-score for one matching dimension (skills,
    experience, or education). The model must ground every score in a
    quoted excerpt from the retrieved resume text — this is the mechanism
    that keeps scoring evidence-based rather than an opaque number.
    """

    score: float = Field(ge=1, le=10, description="Fit score for this dimension, 1-10")
    justification: str = Field(
        description="2-3 sentence explanation referencing the job requirement and what was found"
    )
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Exact short quotes copied from the retrieved resume text that support the score",
    )


class CombinedDimensionScores(BaseModel):
    """All three dimension scores from a single LLM call. Retrieval is
    still done separately per dimension (see scoring.score_all_dimensions)
    so each score is grounded in its own excerpts -- this only consolidates
    the *scoring* call itself, to cut API round-trips.
    """

    skills: DimensionScore
    experience: DimensionScore
    education: DimensionScore


class InterviewQuestion(BaseModel):
    question: str
    targets: str = Field(description="The gap or dimension this question is meant to probe")


class InterviewQuestions(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)
