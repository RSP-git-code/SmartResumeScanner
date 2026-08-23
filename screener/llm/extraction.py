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
        "everything to the same weight.\n\n"
        "Separately, pull out hard pass/fail eligibility cutoffs into "
        "eligibility_criteria, in plain language, using whatever scale the "
        "JD itself uses -- e.g. a minimum academic score (percentage, "
        "CGPA, or otherwise, exactly as stated), a requirement that the "
        "candidate has already completed their degree (as opposed to "
        "currently pursuing it), a mandatory minimum years of experience "
        "presented as a strict cutoff rather than a general expectation, "
        "or a required degree branch/field of study (e.g. a role that "
        "specifically requires an engineering/technical degree like "
        "B.Tech/B.E. in Computer Science -- a candidate whose degree is in "
        "an unrelated field, such as an MBA applying to that role, should "
        "fail this criterion). These are distinct from required_skills/"
        "preferred_skills. Leave the list empty if the JD states no such "
        "hard cutoffs -- do not invent eligibility rules that aren't "
        "explicitly there.\n\n"
        "Only include criteria that are objective, factual, and the kind "
        "of thing a resume would actually state (academic scores, degree "
        "status, degree branch, years of experience). Do NOT include "
        "subjective, attitudinal, or culture-fit language even if the JD "
        "phrases it as a requirement -- e.g. 'willingness to work in a "
        "start-up environment', 'team player', 'passionate about coding', "
        "'excellent communication skills', 'ability to work under "
        "pressure'. Resumes essentially never state these explicitly, and "
        "since a criterion the resume doesn't address is treated as "
        "failed, including a subjective one here would wrongly reject "
        "almost every candidate. When in doubt about whether a stated "
        "requirement is objectively checkable from resume text, leave it "
        "out of eligibility_criteria (it can still be reflected in the "
        "skills/experience scoring instead).",
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
