"""Structured extraction chains: turn raw resume/JD text into the typed
schemas defined in schemas.py, via LangChain `with_structured_output`
rather than asking for free text and hoping it parses as JSON.
"""

from langchain_core.prompts import ChatPromptTemplate

from .client import get_chat_model
from .schemas import JobRequirements, ResumeProfile

RESUME_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract structured data from resumes. Extract only what is "
        "explicitly stated in the text -- do not invent skills, employers, "
        "or dates. Normalize skill names to common industry terms (e.g. "
        "'Postgres' -> 'PostgreSQL') but never add a skill that isn't "
        "mentioned anywhere in the text.",
    ),
    ("human", "Resume text:\n\n{resume_text}"),
])

JD_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You extract structured hiring requirements from a job "
        "description. Separate 'required' from 'preferred/nice-to-have' "
        "skills based on the language used (e.g. 'must have' vs 'a plus', "
        "'preferred'). Only include what is explicitly stated in the "
        "text.\n\n"
        "For each required skill, assign an importance weight from 1-5 "
        "based on how the text emphasizes it: 5 for skills called out as "
        "critical/must-have or mentioned multiple times, 3 for skills "
        "listed as requirements without special emphasis, 1 for skills "
        "mentioned only in passing. This weighting drives a candidate "
        "ranking matrix, so be deliberate about it rather than defaulting "
        "everything to the same weight.",
    ),
    ("human", "Job description text:\n\n{jd_text}"),
])


def extract_resume_profile(resume_text: str) -> ResumeProfile:
    model = get_chat_model().with_structured_output(ResumeProfile)
    chain = RESUME_EXTRACTION_PROMPT | model
    return chain.invoke({"resume_text": resume_text})


def extract_job_requirements(jd_text: str) -> JobRequirements:
    model = get_chat_model().with_structured_output(JobRequirements)
    chain = JD_EXTRACTION_PROMPT | model
    return chain.invoke({"jd_text": jd_text})
