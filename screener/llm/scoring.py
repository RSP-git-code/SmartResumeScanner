"""Multi-dimensional, rubric-constrained scoring plus the deterministic
skill-overlap anchor, blended into the final hybrid score.

Each dimension (skills/experience/education) is grounded independently:
the retriever pulls only the resume excerpts relevant to that dimension.
The LLM call itself is consolidated into a single request that scores all
three dimensions together (see score_all_dimensions) -- retrieval is still
per-dimension so scores stay evidence-grounded, this only cuts API
round-trips, which matters a lot against a tight free-tier rate limit.
The deterministic component is pure set overlap with no LLM judgment
involved, acting as a bias-resistant anchor that the semantic score is
blended against (see redaction.py for the full rationale).
"""

from django.conf import settings
from langchain_core.prompts import ChatPromptTemplate

from .client import get_chat_model
from .schemas import CombinedDimensionScores

COMBINED_SCORING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are scoring how well a candidate matches a job across three "
        "independent dimensions: skills, experience, and education. Score "
        "each 1-10. For each dimension, base the score ONLY on that "
        "dimension's retrieved excerpts below -- if something isn't "
        "mentioned in the excerpts, treat it as absent even if it seems "
        "likely from context. Quote the exact excerpt text you relied on "
        "in each dimension's evidence_quotes. Score the three dimensions "
        "independently -- a weak result in one should not pull down "
        "another.",
    ),
    (
        "human",
        "== SKILLS ==\n"
        "Required skills: {required_skills}\n"
        "Preferred skills: {preferred_skills}\n"
        "Retrieved excerpts:\n{skills_context}\n\n"
        "== EXPERIENCE ==\n"
        "Minimum years of experience required: {min_experience_years}\n"
        "Retrieved excerpts:\n{experience_context}\n\n"
        "== EDUCATION ==\n"
        "Education requirement: {education_requirement}\n"
        "Retrieved excerpts:\n{education_context}",
    ),
])

DIMENSION_QUERIES = {
    "skills": "skills, technologies, tools, programming languages",
    "experience": "work experience, job titles, responsibilities, years of experience",
    "education": "education, degree, university, college",
}


def score_all_dimensions(
    retriever, required_skills, preferred_skills, min_experience_years, education_requirement
) -> CombinedDimensionScores:
    def context_for(dimension):
        chunks = retriever.invoke(DIMENSION_QUERIES[dimension])
        return "\n---\n".join(c.page_content for c in chunks) or "(no relevant excerpts found)"

    model = get_chat_model().with_structured_output(CombinedDimensionScores)
    chain = COMBINED_SCORING_PROMPT | model
    return chain.invoke({
        "required_skills": format_required_skills(required_skills),
        "preferred_skills": ", ".join(preferred_skills) or "none specified",
        "skills_context": context_for("skills"),
        "min_experience_years": min_experience_years,
        "experience_context": context_for("experience"),
        "education_requirement": education_requirement,
        "education_context": context_for("education"),
    })


def format_required_skills(required_skills) -> str:
    """`required_skills` is the list of {"skill": str, "weight": int}
    dicts from JobDescription.extracted_requirements. Renders them for the
    LLM prompt so weight is visible to the model, not just to the
    deterministic scorer.
    """
    if not required_skills:
        return "none specified"
    return ", ".join(f"{s['skill']} (weight {s['weight']}/5)" for s in required_skills)


def deterministic_skill_overlap(resume_skills, required_skills, preferred_skills) -> float:
    """Weighted, pure set-overlap score (0-10), no LLM judgment involved.

    `required_skills` is the list of {"skill": str, "weight": int 1-5}
    dicts extracted from the JD -- missing a weight-5 "critical" skill
    costs far more than missing a weight-1 one. Required skills are
    weighted 80% of the score, flat-weighted preferred skills 20%, mirroring
    the semantic scorer's treatment of required vs. nice-to-have.
    """

    def norm(items):
        return {s.strip().lower() for s in items if s and s.strip()}

    resume_set = norm(resume_skills)
    preferred_set = norm(preferred_skills)
    total_weight = sum(entry.get('weight', 3) for entry in required_skills)

    if total_weight == 0 and not preferred_set:
        return 5.0  # no explicit requirements to compare against

    if total_weight:
        matched_weight = sum(
            entry.get('weight', 3)
            for entry in required_skills
            if str(entry.get('skill', '')).strip().lower() in resume_set
        )
        required_ratio = matched_weight / total_weight
    else:
        required_ratio = 1.0

    preferred_ratio = (len(resume_set & preferred_set) / len(preferred_set)) if preferred_set else 0.0

    combined = 0.8 * required_ratio + 0.2 * preferred_ratio
    return round(combined * 10, 2)


def combine_scores(llm_semantic_score: float, deterministic_score: float) -> float:
    weighted = (
        settings.LLM_SCORE_WEIGHT * llm_semantic_score
        + settings.DETERMINISTIC_SCORE_WEIGHT * deterministic_score
    )
    return round(weighted, 2)
