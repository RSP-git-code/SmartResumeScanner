"""Central place to construct LangChain model clients from Django settings,
so nothing else in the pipeline hardcodes a model name or API key.

Also wires up on-disk response caching: every chain in this project calls
the model at temperature=0, so identical prompts (e.g. re-running the
pipeline against the same sample_data/ fixtures while iterating) return an
instant, free, cached response instead of a fresh billed API call. This
also means repeatedly testing during development won't burn through rate
limits the way uncached re-runs would.
"""

from django.conf import settings
from langchain_core.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Transient, worth-retrying failure categories from langchain_core's
# provider-agnostic error taxonomy -- e.g. Gemini's GoogleRateLimitError
# (429) and GoogleAPIError (503 "high demand") are subclasses of
# ModelRateLimitError/ModelAPIError respectively. Retrying by this taxonomy
# rather than a specific provider's exception class means a new transient
# error subtype from Gemini (or a future provider) is covered automatically,
# instead of needing a code change every time Google adds one. Deliberately
# excludes non-transient ModelError subclasses (auth, permission, invalid
# request, model-not-found, context-overflow) -- retrying those can't help,
# they'll fail identically every time.
TRANSIENT_MODEL_ERRORS = (ModelRateLimitError, ModelAPIError, ModelConnectionError, ModelTimeoutError)

set_llm_cache(SQLiteCache(database_path=str(settings.BASE_DIR / "llm_cache.db")))


def get_chat_model(temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
        # Kept tight on purpose: the underlying google-genai SDK has its own
        # internal retry loop (separate from with_llm_retry below), so
        # max_retries=3 x timeout=60 meant a single slow/hanging call could
        # block for up to 180s *before* with_llm_retry (which only kicks in
        # once the SDK finally raises) ever got a chance to run -- enough on
        # its own to blow through run_matching's per-request budget across
        # several resumes. A stuck call should fail fast so retry logic
        # actually gets to act on it.
        max_retries=1,
        timeout=30,
    )


def with_llm_retry(chain):
    """Wraps a chain with backoff retry for transient provider failures --
    Gemini's free-tier per-minute rate limit (429), momentary server
    overload (503 "high demand"), connection hiccups, timeouts -- easy to
    hit when matching several resumes in one request, since each resume
    makes several LLM calls back to back (extraction, redaction, scoring,
    interview questions). ChatGoogleGenerativeAI's own `max_retries` isn't
    enough here since it doesn't wait long enough for a per-minute quota to
    reset; this backs off for a few seconds between attempts instead of
    failing that resume's match outright on the first transient error.

    Kept deliberately tight (2 retries, capped backoff) rather than
    generous: `run_matching` runs this per resume, several resumes deep in
    one request, all under gunicorn's own request timeout -- a longer
    per-call retry budget compounds across resumes and can blow through
    that timeout, killing the whole batch instead of just failing one
    resume's match (worse than the plain error this is meant to soften).
    """
    return chain.with_retry(
        retry_if_exception_type=TRANSIENT_MODEL_ERRORS,
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 3, "max": 10},
        stop_after_attempt=3,
    )


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=settings.GEMINI_EMBEDDING_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
    )


def message_text(message) -> str:
    """AIMessage.content is a plain string for most providers, but Gemini
    returns a list of content blocks (e.g. [{"type": "text", "text": ...}]).
    Chains that use with_structured_output are unaffected by this (parsing
    happens at the tool-calling layer); this helper is only needed by
    chains that read raw message text, like redaction.py.
    """
    content = message.content
    if isinstance(content, str):
        return content.strip()

    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts).strip()
