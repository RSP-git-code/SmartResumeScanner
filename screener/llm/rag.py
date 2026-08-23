"""Chunk + embed redacted resume text into a per resume/JD Chroma
collection and expose a retriever, so scoring is grounded in retrieved
excerpts instead of the whole resume being stuffed into one prompt.
"""

from django.conf import settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .client import get_embeddings

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)


def _collection_name(resume_id: int, job_description_id: int) -> str:
    return f"resume_{resume_id}_jd_{job_description_id}"


def build_resume_retriever(resume_id: int, job_description_id: int, redacted_text: str, k: int = 4):
    collection_name = _collection_name(resume_id, job_description_id)
    embeddings = get_embeddings()

    # Drop any embeddings from a previous run of this exact pair first, so
    # re-matching doesn't silently accumulate duplicate chunks.
    Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    ).delete_collection()

    docs = [Document(page_content=chunk) for chunk in _SPLITTER.split_text(redacted_text)]
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})
