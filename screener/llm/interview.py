"""Bonus chain: turns the weakest-scoring dimensions and their
justifications into targeted interview questions, so a shortlisted
candidate's report is a decision-support artifact and not just a number.
"""

from langchain_core.prompts import ChatPromptTemplate

from .client import get_chat_model
from .schemas import InterviewQuestions

INTERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You generate targeted interview questions for a candidate being "
        "considered for a role. Focus on probing the weakest-scoring "
        "dimensions and the specific gaps called out in their "
        "justifications -- these are what an interviewer most needs to "
        "verify in person. Generate 3-5 questions. Each must be specific "
        "enough to actually test the gap, not generic ('tell me about "
        "yourself').",
    ),
    (
        "human",
        "Job title: {job_title}\n\n"
        "Skills -- score {skills_score}/10: {skills_justification}\n"
        "Experience -- score {experience_score}/10: {experience_justification}\n"
        "Education -- score {education_score}/10: {education_justification}\n",
    ),
])


def generate_interview_questions(
    job_title,
    skills_score,
    skills_justification,
    experience_score,
    experience_justification,
    education_score,
    education_justification,
):
    model = get_chat_model().with_structured_output(InterviewQuestions)
    chain = INTERVIEW_PROMPT | model
    result = chain.invoke({
        "job_title": job_title,
        "skills_score": skills_score,
        "skills_justification": skills_justification,
        "experience_score": experience_score,
        "experience_justification": experience_justification,
        "education_score": education_score,
        "education_justification": education_justification,
    })
    return [{"question": q.question, "targets": q.targets} for q in result.questions]
