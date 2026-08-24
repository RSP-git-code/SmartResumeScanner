"""PII redaction: the bias-mitigation step of the pipeline.

Runs before any scoring call so the scoring chain (screener/llm/scoring.py)
only ever sees resume text with direct identity/demographic markers
stripped out. This is deliberately narrow in scope -- it removes what it
can reliably catch (name, contact info, address, age, gendered
honorifics/pronouns, marital/nationality/religion mentions) and leaves
everything relevant to job fit untouched (skills, employers, titles,
durations, education degree/institution/field).

Known limitation (documented for the README): this does not neutralize
indirect signals such as institution prestige or hobby choices -- a true
blind-review comparison wouldn't either, since those signals aren't tied to
an explicit field that can be stripped. The mitigation here is combined
with rubric-constrained, evidence-quoting prompts (scoring.py) and a
deterministic non-LLM anchor score to limit how much any single biased
judgment can sway the final score.
"""

from langchain_core.prompts import ChatPromptTemplate

from .client import get_chat_model, message_text, with_llm_retry

REDACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite resume text to remove personally-identifying and "
        "demographic-signaling details before it is used for automated "
        "job-fit scoring, so scoring cannot be swayed by attributes "
        "irrelevant to job fit.\n\n"
        "Replace with the bracketed placeholder shown; remove entirely "
        "where no placeholder is given. Keep everything else verbatim -- "
        "skills, job titles, employers, employment durations, "
        "responsibilities, and education degree/institution/field must "
        "all be preserved exactly:\n"
        "- Full name -> [CANDIDATE]\n"
        "- Street address / city / country -> [LOCATION]\n"
        "- Age or date of birth -> [AGE]\n"
        "- Gendered honorifics (Mr./Ms./Mrs.) or pronouns -> remove\n"
        "- Marital status, nationality, religion, photo references -> remove\n"
        "- Personal email/phone number -> [CONTACT]\n\n"
        "Do not summarize, do not drop any professional content, do not "
        "add commentary or notes. Return only the rewritten resume text.",
    ),
    ("human", "{resume_text}"),
])


def redact_resume_text(resume_text: str) -> str:
    model = get_chat_model()
    chain = with_llm_retry(REDACTION_PROMPT | model)
    result = chain.invoke({"resume_text": resume_text})
    return message_text(result)
