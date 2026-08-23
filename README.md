# Smart Resume Screener

Parses PDF/text resumes and a job description, extracts structured data, and produces a **ranked, evidence-grounded, bias-mitigated shortlist** with per-candidate interview questions — built for the Unthinkable Smart Resume Screener assignment.

## Why this isn't just "stuff resume + JD into one prompt"

The brief's example prompt (*"Compare the following resume with this job description and rate fit 1-10 with justification"*) is a reasonable baseline, but it has three problems this project deliberately avoids:

1. **No evidence trail.** A single opaque score is hard to trust or audit. Every score here is grounded in retrieved resume excerpts the model must quote, and the full justification is stored, not just displayed.
2. **No structure.** One "fit" number hides *what* it's measuring. Scoring here is split into Skills / Experience / Education, each independently scored and justified, plus an actual **skill × candidate ranking matrix** driven by per-skill importance weights extracted from the JD.
3. **Bias risk.** An LLM judging "fit" from a full resume can be swayed by name, age, gender markers, or institution prestige — signals that have nothing to do with job fit. This pipeline redacts identity/demographic markers before any scoring call, constrains scoring to cited evidence only, and blends the LLM's semantic judgment with a deterministic, non-LLM skill-overlap score so no single biased judgment can dominate the final result.
4. **Wasted good candidates.** A candidate who's a poor fit for the role they applied to might be a strong fit for a *different* open role — real recruiting teams call this internal redeployment/talent pooling. Since resumes here are a shared pool matchable against any job description (not scoped to one posting), each shortlisted candidate has a **"Check other roles"** action that scores them against every other open JD and flags it when they'd score meaningfully higher elsewhere.

## Architecture

```
                 ┌─────────────────────────────────────────────────────┐
                 │                    Django project                    │
                 │                                                       │
  Upload/API ───▶│  screener/api.py (DRF)   screener/views.py (HTML)    │
                 │            │                       │                 │
                 │            └───────────┬───────────┘                 │
                 │                        ▼                              │
                 │           screener/llm/pipeline.py                   │
                 │           (single orchestration entry point)          │
                 │                        │                              │
                 │   ┌────────────────────┼────────────────────┐        │
                 │   ▼                    ▼                    ▼        │
                 │ extraction.py     redaction.py          rag.py       │
                 │ (structured        (strip PII           (chunk +     │
                 │  skills/exp/edu     before scoring)      embed into  │
                 │  via structured                           Chroma,    │
                 │  output)                                  per        │
                 │                                            resume/JD)│
                 │                        │                    │        │
                 │                        ▼                    ▼        │
                 │                  scoring.py ◀────────────────┘       │
                 │            (rubric-constrained, evidence-quoting     │
                 │             sub-scores + deterministic weighted      │
                 │             skill-overlap anchor + hybrid blend)     │
                 │                        │                              │
                 │                        ▼                              │
                 │                 interview.py                        │
                 │        (targets weakest-scoring dimensions)          │
                 │                        │                              │
                 │                        ▼                              │
                 │              screener/models.py (Match)              │
                 └─────────────────────────────────────────────────────┘
```

- **Backend**: Django 5 + Django REST Framework
- **LLM orchestration**: LangChain (`langchain`, `langchain-google-genai`, `langchain-chroma`)
- **LLM provider**: Google Gemini — chat model and embedding model are both plain env vars (`GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL` in `.env`), nothing hardcoded
- **Vector store**: Chroma, persisted locally to `chroma_db/`, one collection per resume/JD pair
- **PDF parsing**: `pdfplumber`
- **Database**: SQLite (fine at this scale; swappable via `DATABASES` in `resume_screener/settings.py`)
- **Frontend**: Django templates + Bootstrap 5 (server-rendered dashboard, no separate frontend build)
- **Structured LLM output**: Pydantic schemas (`screener/llm/schemas.py`) enforced via LangChain's `with_structured_output`, not free-text parsing

## The pipeline, step by step

For each resume matched against a job description (`screener/llm/pipeline.py:run_match`):

1. **Structured extraction** (`extraction.py`) — resume → `{candidate_name, skills[], experience[], education[]}`; JD → `{required_skills: [{skill, weight 1-5}], preferred_skills[], min_experience_years, education_requirement}`. The weight per required skill is what drives the ranking matrix and the deterministic scorer below.
2. **PII redaction** (`redaction.py`) — the *bias-mitigation step*. Runs before anything else touches the resume text for scoring. Strips name, address, age, gendered honorifics/pronouns, marital/nationality/religion mentions, and personal contact info, replacing with placeholders (`[CANDIDATE]`, `[LOCATION]`, etc.) while preserving every word of professional content (skills, titles, employers, durations, education). Only the redacted text is ever sent to the scoring chain.
3. **RAG indexing** (`rag.py`) — the redacted resume text is chunked (`RecursiveCharacterTextSplitter`, 500 chars/75 overlap) and embedded into a Chroma collection scoped to this resume+JD pair. A fresh retriever is built per match (stale embeddings from a prior run of the same pair are dropped first).
4. **Multi-dimensional scoring** (`scoring.py`) — for each of Skills / Experience / Education, the retriever pulls only the excerpts relevant to that dimension (three separate retrieval queries), and a single LLM call scores all three together, required to quote the exact excerpt it relied on for each. If a skill isn't in the retrieved excerpts, the model is instructed to treat it as absent — no reasoning from the full resume in memory.
5. **Deterministic anchor score** (`scoring.deterministic_skill_overlap`) — pure Python set overlap between the candidate's extracted skills and the JD's weighted required/preferred skills. No LLM involved. Missing a weight-5 "critical" skill costs far more than missing a weight-1 one. This is what renders as the **skill ranking matrix** on the dashboard.
6. **Hybrid final score** — `final_score = LLM_SCORE_WEIGHT × llm_semantic_score + DETERMINISTIC_SCORE_WEIGHT × deterministic_score` (defaults 0.6/0.4, configurable in `.env`). The deterministic half acts as a bias-resistant anchor the semantic score is blended against, rather than trusting the LLM's judgment alone.
7. **Interview questions** (`interview.py`) — 3-5 questions targeting the weakest-scoring dimensions and the specific gaps named in their justifications, so a shortlisted candidate's entry is a decision-support artifact, not just a number.

Separately, on demand per candidate (`views.check_other_roles` / `api.check_other_roles`, not part of the automatic per-JD flow above since cost scales with the number of open JDs):

8. **Cross-role recommendation** — re-runs steps 3-6 for the same resume against every *other* open `JobDescription`, and if any of them scores higher than the role currently being viewed, surfaces "may be a stronger fit for `<other role>`" on the dashboard. Reuses the exact same `run_match` pipeline and `Match` model — no separate schema, since resumes were already a shared pool matchable against any JD.

**Known limitation**: redaction only catches direct identity markers. It doesn't neutralize indirect signals like institution prestige or hobby choices — a true blind-review process wouldn't fully solve that either, since those signals aren't tied to a strippable field. The rubric-constrained, evidence-quoting prompts and the deterministic anchor score are what limit how much any single biased judgment can sway the outcome, rather than claiming redaction alone solves bias.

## LLM prompts

All prompts live in `screener/llm/`; reproduced here per the assignment's documentation requirement.

**Resume extraction** (`extraction.py`, `RESUME_EXTRACTION_PROMPT`):
> You extract structured data from resumes. Extract only what is explicitly stated in the text -- do not invent skills, employers, or dates. Normalize skill names to common industry terms (e.g. 'Postgres' -> 'PostgreSQL') but never add a skill that isn't mentioned anywhere in the text.

**JD requirement extraction** (`extraction.py`, `JD_EXTRACTION_PROMPT`):
> You extract structured hiring requirements from a job description. Separate 'required' from 'preferred/nice-to-have' skills based on the language used (e.g. 'must have' vs 'a plus', 'preferred'). Only include what is explicitly stated in the text.
>
> For each required skill, assign an importance weight from 1-5 based on how the text emphasizes it: 5 for skills called out as critical/must-have or mentioned multiple times, 3 for skills listed as requirements without special emphasis, 1 for skills mentioned only in passing. This weighting drives a candidate ranking matrix, so be deliberate about it rather than defaulting everything to the same weight.

**PII redaction** (`redaction.py`, `REDACTION_PROMPT`):
> You rewrite resume text to remove personally-identifying and demographic-signaling details before it is used for automated job-fit scoring, so scoring cannot be swayed by attributes irrelevant to job fit.
>
> Replace with the bracketed placeholder shown; remove entirely where no placeholder is given. Keep everything else verbatim -- skills, job titles, employers, employment durations, responsibilities, and education degree/institution/field must all be preserved exactly:
> - Full name -> [CANDIDATE]
> - Street address / city / country -> [LOCATION]
> - Age or date of birth -> [AGE]
> - Gendered honorifics (Mr./Ms./Mrs.) or pronouns -> remove
> - Marital status, nationality, religion, photo references -> remove
> - Personal email/phone number -> [CONTACT]
>
> Do not summarize, do not drop any professional content, do not add commentary or notes. Return only the rewritten resume text.

**Multi-dimensional scoring** (`scoring.py`, `COMBINED_SCORING_PROMPT`):
> You are scoring how well a candidate matches a job across three independent dimensions: skills, experience, and education. Score each 1-10. For each dimension, base the score ONLY on that dimension's retrieved excerpts below -- if something isn't mentioned in the excerpts, treat it as absent even if it seems likely from context. Quote the exact excerpt text you relied on in each dimension's evidence_quotes. Score the three dimensions independently -- a weak result in one should not pull down another.
>
> *(human message fills in required/preferred skills, min experience years, education requirement, and the three sets of per-dimension retrieved excerpts)*

**Interview question generation** (`interview.py`, `INTERVIEW_PROMPT`):
> You generate targeted interview questions for a candidate being considered for a role. Focus on probing the weakest-scoring dimensions and the specific gaps called out in their justifications -- these are what an interviewer most needs to verify in person. Generate 3-5 questions. Each must be specific enough to actually test the gap, not generic ('tell me about yourself').

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows; source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

copy .env.example .env           # cp on macOS/Linux
# then edit .env: set GOOGLE_API_KEY, and GEMINI_MODEL/GEMINI_EMBEDDING_MODEL if not using the defaults

python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`, create a job description, upload resumes, click "Run Matching".

### Verifying the pipeline without the UI

```bash
python manage.py smoke_test
```

Runs the full extract → redact → RAG → score → interview pipeline against the three fixtures in `sample_data/` (a JD + a strong/partial/weak-fit resume each) and prints the resulting ranking. Idempotent — safe to re-run, cleans up its own prior output first.

### Running tests

```bash
python manage.py test
```

Covers model behavior, PDF/text extraction, and the deterministic weighted-scoring math — everything that doesn't require a live LLM call.

## Deploying to Render

The codebase is deploy-ready (gunicorn, whitenoise for static files, `DATABASE_URL` support via `dj-database-url`), but creating the actual Render services needs your account/login, so these are manual steps:

1. Push the repo to GitHub (Render deploys from a connected repo).
2. On [Render](https://dashboard.render.com), **New +** → **Web Service** → connect the repo.
3. Runtime: Python. Build command:
   ```
   pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate
   ```
   Start command:
   ```
   gunicorn resume_screener.wsgi:application
   ```
4. Set environment variables on the service: `DJANGO_SECRET_KEY` (any random string), `DEBUG=False`, `GOOGLE_API_KEY`, and optionally `GEMINI_MODEL`/`GEMINI_EMBEDDING_MODEL`/`LLM_SCORE_WEIGHT`/`DETERMINISTIC_SCORE_WEIGHT` if not using the defaults. `ALLOWED_HOSTS` and CSRF are handled automatically — `resume_screener/settings.py` reads Render's own `RENDER_EXTERNAL_HOSTNAME` env var, which Render injects for you.
5. Deploy. First load after the build finishes may take a minute.

**Two free-tier limitations worth knowing before you rely on this for anything long-lived:**

- **No persistent disk on free web services.** SQLite (the default `DATABASE_URL`-less config) and any uploaded resume files live on the service's local disk, which resets whenever the free service sleeps from inactivity and wakes back up — meaning all data can vanish between visits. Fine for demoing in one sitting right after a deploy; not fine as a durable link to share.
- **To survive sleep/wake cycles**, attach a Render PostgreSQL instance (**New +** → **PostgreSQL**, free tier available) and copy its Internal Database URL into the web service's `DATABASE_URL` env var — no code changes needed, `dj-database-url` picks it up automatically. But free Render Postgres instances **expire 30 days after creation**, so this is still not a permanent fix, just a longer-lived one. Uploaded resume *files* (as opposed to the extracted text, which lives in the database either way) still don't persist even with Postgres, since `MEDIA_ROOT` is on the same ephemeral web-service disk — would need external object storage (e.g. S3) to fix that, out of scope here.
- Free web services also spin down after inactivity and take ~30-60s to wake on the next request — the first hit after a while looks slow, that's expected.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET/POST` | `/api/job-descriptions/` | List JDs / create one (`title` + `text` or `file`), triggers requirement extraction |
| `POST` | `/api/resumes/upload/` | Upload one or more resumes (`files`), triggers structured extraction + redaction |
| `POST` | `/api/match/` | `{job_description_id, resume_ids: []}` (omit `resume_ids` to match all resumes) — runs the full pipeline per pair |
| `GET` | `/api/job-descriptions/<id>/shortlist/` | Ranked matches with full score breakdown, evidence, and interview questions |
| `POST` | `/api/resumes/<id>/check-other-roles/` | `{exclude_job_description_id}` (optional) — scores this candidate against every other open JD, for the "may fit better elsewhere" recommendation |

The dashboard (`/`, `/jobs/<id>/`) is a thin HTML layer over the exact same `screener/llm/pipeline.py` functions the API calls — no duplicated business logic between the two.

## Configuration reference (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | — | Required. Gemini API key. |
| `GEMINI_MODEL` | `gemini-flash-lite-latest` | Chat model for all extraction/redaction/scoring/interview calls. |
| `GEMINI_EMBEDDING_MODEL` | `models/gemini-embedding-001` | Embedding model for RAG indexing. |
| `LLM_SCORE_WEIGHT` / `DETERMINISTIC_SCORE_WEIGHT` | `0.6` / `0.4` | Hybrid score blend weights (must sum to 1.0). |
| `DJANGO_SECRET_KEY` | (dev-only fallback) | Set a real random value in production. |
| `DEBUG` | `True` | Set to `False` in production. |
| `ALLOWED_HOSTS` | (empty) | Comma-separated extra hosts. Render's own domain is auto-detected via `RENDER_EXTERNAL_HOSTNAME`, no need to set this for a Render deploy specifically. |
| `DATABASE_URL` | (unset → SQLite) | Set to a Postgres URL (e.g. from a Render Postgres instance) for a database that survives web-service sleep/wake cycles. |

## Known constraints

- **Free-tier quota varies a lot by model.** `gemini-3.6-flash` (the newest preview at time of writing) capped at a hard 20 requests/day on the free tier — with 4 LLM calls per resume match (extraction, redaction, combined scoring, interview questions) plus 1 for the JD, that's barely 4 resumes/day. Switched the default to `gemini-flash-lite-latest`, an alias Google keeps pointed at their current lite-tier flash model, which comfortably handled a full 3-resume smoke test (13 calls) with no rate-limit errors. `GEMINI_MODEL` is a plain env var, so swapping models (or enabling billing for higher throughput) needs no code changes. `screener/llm/client.py` also wires up SQLite-backed response caching (`llm_cache.db`) so re-running the same prompts during development doesn't re-consume quota either way.
- **Synchronous pipeline**: `run_match` makes multiple LLM calls in-line within the request. Fine at this scale; a production version would move it onto a task queue (Celery) so uploads/matching don't block on network round-trips.
- **Some Gemini models ignore the `temperature` parameter** (fixed sampling defaults, e.g. observed on `gemini-3.6-flash`) — cosmetic warning, doesn't affect correctness, but means run-to-run determinism on those models relies on the model's own consistency rather than our `temperature=0` setting.
