"""Raw text extraction from uploaded resume/JD files (PDF or plain text).

Kept dependency-free of LangChain/Django models on purpose: this is a pure
"bytes/file -> text" utility so it can be unit tested and reused by both the
upload views and one-off scripts (e.g. sample_data smoke tests).
"""

from pathlib import Path

import pdfplumber


def extract_text(file_obj, filename=None):
    """Extract text from a PDF or plain-text file.

    `file_obj` may be a Django UploadedFile, a file-like object opened in
    binary mode, or a path (str/Path). `filename` overrides the name used to
    decide the extraction strategy (needed for in-memory file-like objects
    whose `.name` isn't reliable).
    """
    name = filename or getattr(file_obj, 'name', None) or str(file_obj)
    suffix = Path(name).suffix.lower()

    if suffix == '.pdf':
        return _extract_pdf_text(file_obj)
    return _extract_plain_text(file_obj)


def _extract_pdf_text(file_obj):
    with pdfplumber.open(file_obj) as pdf:
        pages = [page.extract_text() or '' for page in pdf.pages]

    text = '\n\n'.join(pages).strip()
    if not text:
        raise ValueError(
            'No extractable text found in PDF — it may be a scanned image '
            'without a text layer.'
        )
    return text


def _extract_plain_text(file_obj):
    if hasattr(file_obj, 'read'):
        raw = file_obj.read()
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
    else:
        raw = Path(file_obj).read_bytes()

    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='replace')

    text = raw.strip()
    if not text:
        raise ValueError('No text found in file.')
    return text
