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
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import GoogleRateLimitError

set_llm_cache(SQLiteCache(database_path=str(settings.BASE_DIR / "llm_cache.db")))


def get_chat_model(temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
        max_retries=3,
        timeout=60,
    )


def with_llm_retry(chain):
    """Wraps a chain with backoff retry for Gemini's free-tier per-minute
    rate limit (GoogleRateLimitError / 429 RESOURCE_EXHAUSTED) -- easy to
    hit when matching several resumes in one request, since each resume
    makes several LLM calls back to back (extraction, redaction, scoring,
    interview questions). ChatGoogleGenerativeAI's own `max_retries` isn't
    enough here since it doesn't wait long enough for a per-minute quota to
    reset; this backs off for several seconds between attempts instead of
    failing that resume's match outright on the first 429.
    """
    return chain.with_retry(
        retry_if_exception_type=(GoogleRateLimitError,),
        wait_exponential_jitter=True,
        exponential_jitter_params={"initial": 5, "max": 20},
        stop_after_attempt=4,
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
