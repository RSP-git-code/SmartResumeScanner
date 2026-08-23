"""Runs the full extract/redact/RAG/score/interview pipeline against
sample_data/ fixtures, without going through the UI or API. Re-runnable --
cleans up its own previous output first, identified by a reserved JD title.

Useful for verifying the pipeline end-to-end after a provider, prompt, or
scoring-weight change without manually clicking through the dashboard.
"""

import io

from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand

from screener import text_extraction
from screener.llm import pipeline
from screener.models import JobDescription, Resume

SAMPLE_DIR = settings.BASE_DIR / 'sample_data'
JD_FILE = SAMPLE_DIR / 'job_description_backend_engineer.pdf'
RESUME_FILES = [
    ('Strong match', SAMPLE_DIR / 'resume_strong_match.pdf'),
    ('Partial match', SAMPLE_DIR / 'resume_partial_match.pdf'),
    ('Weak match', SAMPLE_DIR / 'resume_weak_match.pdf'),
]
JD_TITLE = '[smoke test] Backend Engineer - Python'


class Command(BaseCommand):
    help = 'Exercises the full LLM pipeline against sample_data/ fixtures (no UI/API involved).'

    def handle(self, *args, **options):
        old_jd = JobDescription.objects.filter(title=JD_TITLE).first()
        if old_jd:
            Resume.objects.filter(matches__job_description=old_jd).delete()
            old_jd.delete()

        self.stdout.write('Creating job description...')
        jd_text = text_extraction.extract_text(JD_FILE)
        jd = JobDescription.objects.create(title=JD_TITLE, raw_text=jd_text)
        pipeline.process_job_description(jd)
        self.stdout.write(self.style.SUCCESS(f'  requirements: {jd.extracted_requirements}'))

        results = []
        for label, path in RESUME_FILES:
            self.stdout.write(f'Processing resume: {label} ({path.name})...')
            raw_bytes = path.read_bytes()
            resume_text = text_extraction.extract_text(io.BytesIO(raw_bytes), filename=path.name)

            resume = Resume(raw_text=resume_text)
            resume.file.save(path.name, ContentFile(raw_bytes), save=True)

            pipeline.process_resume(resume)
            match = pipeline.run_match(resume, jd)
            results.append((label, match))
            self.stdout.write(self.style.SUCCESS(
                f'  {resume.candidate_name}: final={match.final_score} '
                f'(llm={match.llm_semantic_score}, rule={match.deterministic_score})'
            ))

        self.stdout.write('\nRanking (highest first):')
        results.sort(key=lambda r: r[1].final_score, reverse=True)
        for label, match in results:
            self.stdout.write(f'  {match.final_score:5.2f}  {label} -- {match.resume.candidate_name}')

        expected_order = ['Strong match', 'Partial match', 'Weak match']
        actual_order = [label for label, _ in results]
        if actual_order == expected_order:
            self.stdout.write(self.style.SUCCESS('\nRanking order matches expectation.'))
        else:
            self.stdout.write(self.style.WARNING(
                f'\nRanking order {actual_order} does not match expected {expected_order} '
                '-- inspect scores above.'
            ))
