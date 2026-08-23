import io
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError, transaction
from django.test import SimpleTestCase, TestCase, override_settings

from screener import text_extraction
from screener.llm.scoring import combine_scores, deterministic_skill_overlap, format_required_skills
from screener.models import JobDescription, Match, Resume


class ModelTests(TestCase):
    def test_job_description_defaults(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        self.assertEqual(jd.extracted_requirements, {})

    def test_resume_defaults(self):
        resume = Resume.objects.create(file='resumes/test.pdf', raw_text='...')
        self.assertEqual(resume.extracted_skills, [])
        self.assertEqual(resume.extracted_experience, [])
        self.assertEqual(resume.extracted_education, [])

    def test_match_unique_per_resume_and_job(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        resume = Resume.objects.create(file='resumes/test.pdf', raw_text='...')
        Match.objects.create(resume=resume, job_description=jd, final_score=7.5)

        with self.assertRaises(IntegrityError):
            with transaction.atomic():
                Match.objects.create(resume=resume, job_description=jd, final_score=3.0)

    def test_matches_ordered_by_final_score_descending(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        weak = Resume.objects.create(file='resumes/weak.pdf', raw_text='...')
        strong = Resume.objects.create(file='resumes/strong.pdf', raw_text='...')
        Match.objects.create(resume=weak, job_description=jd, final_score=2.0)
        Match.objects.create(resume=strong, job_description=jd, final_score=9.0)

        ordered = list(jd.matches.values_list('final_score', flat=True))
        self.assertEqual(ordered, [9.0, 2.0])


class CrossRoleRecommendationTests(TestCase):
    """job_detail's "better fit elsewhere" computation -- pure ORM/view
    logic, no LLM call involved, so this is testable without hitting the
    API. See screener/views.py:job_detail and check_other_roles.
    """

    def test_flags_better_scoring_role_elsewhere(self):
        devops_role = JobDescription.objects.create(title='DevOps Engineer', raw_text='...')
        data_role = JobDescription.objects.create(title='Data Analyst', raw_text='...')
        resume = Resume.objects.create(file='resumes/test.pdf', raw_text='...', candidate_name='Test Candidate')

        Match.objects.create(resume=resume, job_description=devops_role, final_score=4.0)
        Match.objects.create(resume=resume, job_description=data_role, final_score=8.5)

        response = self.client.get(f'/jobs/{devops_role.id}/')

        self.assertEqual(response.status_code, 200)
        [match] = response.context['matches']
        self.assertIsNotNone(match.better_fit_elsewhere)
        self.assertEqual(match.better_fit_elsewhere.job_description, data_role)
        self.assertContains(response, 'Data Analyst')
        self.assertContains(response, 'stronger fit')

    def test_no_flag_when_current_role_scores_highest(self):
        devops_role = JobDescription.objects.create(title='DevOps Engineer', raw_text='...')
        data_role = JobDescription.objects.create(title='Data Analyst', raw_text='...')
        resume = Resume.objects.create(file='resumes/test.pdf', raw_text='...', candidate_name='Test Candidate')

        Match.objects.create(resume=resume, job_description=devops_role, final_score=9.0)
        Match.objects.create(resume=resume, job_description=data_role, final_score=3.0)

        response = self.client.get(f'/jobs/{devops_role.id}/')

        [match] = response.context['matches']
        self.assertIsNone(match.better_fit_elsewhere)
        self.assertNotContains(response, 'stronger fit')


class EligibilityGateTests(TestCase):
    """job_detail excludes ineligible matches from the main Shortlist and
    shows them in a separate section instead (screener/views.py:job_detail).
    Pure ORM/view test, no LLM call needed -- eligibility outcome is set
    directly on the Match fixture.
    """

    def test_ineligible_excluded_from_shortlist_shown_separately(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        eligible_resume = Resume.objects.create(
            file='resumes/a.pdf', raw_text='...', candidate_name='Eligible Candidate'
        )
        ineligible_resume = Resume.objects.create(
            file='resumes/b.pdf', raw_text='...', candidate_name='Ineligible Candidate'
        )
        Match.objects.create(resume=eligible_resume, job_description=jd, final_score=8.0, is_eligible=True)
        Match.objects.create(
            resume=ineligible_resume,
            job_description=jd,
            final_score=9.0,
            is_eligible=False,
            eligibility_details=[
                {'criterion': 'Minimum 65% aggregate', 'met': False, 'evidence': 'not stated in resume'},
            ],
        )

        response = self.client.get(f'/jobs/{jd.id}/')

        self.assertEqual([m.resume for m in response.context['matches']], [eligible_resume])
        self.assertEqual([m.resume for m in response.context['ineligible_matches']], [ineligible_resume])
        self.assertContains(response, 'Not Eligible')
        self.assertContains(response, 'Minimum 65% aggregate')

    def test_no_ineligible_section_when_all_eligible(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Candidate')
        Match.objects.create(resume=resume, job_description=jd, final_score=8.0, is_eligible=True)

        response = self.client.get(f'/jobs/{jd.id}/')

        self.assertNotContains(response, 'Not Eligible')


class ResumeDedupTests(TestCase):
    """Uploading the same resume content twice must not create two Resume
    rows (screener/views.py:upload_resumes). Mocks pipeline.process_resume
    since real extraction needs a live LLM call -- this test only covers the
    content-hash dedup check that runs before that call.
    """

    @patch('screener.views.pipeline.process_resume')
    def test_duplicate_content_is_skipped(self, mock_process_resume):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        content = b'Ananya Verma\nPython, Django, PostgreSQL'

        self.client.post(
            f'/jobs/{jd.id}/upload-resumes/',
            {'resumes': [SimpleUploadedFile('resume.txt', content)]},
        )
        self.client.post(
            f'/jobs/{jd.id}/upload-resumes/',
            {'resumes': [SimpleUploadedFile('resume_copy.txt', content)]},
        )

        self.assertEqual(Resume.objects.count(), 1)
        self.assertEqual(mock_process_resume.call_count, 1)


class TextExtractionTests(SimpleTestCase):
    def test_plain_text_extraction(self):
        text = text_extraction.extract_text(io.BytesIO(b'Python, Django, PostgreSQL'), filename='resume.txt')
        self.assertEqual(text, 'Python, Django, PostgreSQL')

    def test_empty_text_raises(self):
        with self.assertRaises(ValueError):
            text_extraction.extract_text(io.BytesIO(b'   '), filename='resume.txt')

    def test_sample_pdf_extracts_nonempty_text(self):
        from django.conf import settings

        pdf_path = settings.BASE_DIR / 'sample_data' / 'resume_strong_match.pdf'
        text = text_extraction.extract_text(pdf_path)
        self.assertIn('Ananya Verma', text)


class ScoringTests(SimpleTestCase):
    REQUIRED = [
        {'skill': 'Python', 'weight': 5},
        {'skill': 'Django', 'weight': 5},
        {'skill': 'PostgreSQL', 'weight': 3},
        {'skill': 'Docker', 'weight': 3},
        {'skill': 'Redis', 'weight': 1},
    ]
    PREFERRED = ['AWS', 'Celery']

    def test_format_required_skills(self):
        self.assertEqual(format_required_skills([]), 'none specified')
        self.assertEqual(format_required_skills(self.REQUIRED[:1]), 'Python (weight 5/5)')

    def test_full_overlap_scores_ten(self):
        candidate_skills = ['Python', 'Django', 'PostgreSQL', 'Docker', 'Redis', 'AWS', 'Celery']
        self.assertEqual(deterministic_skill_overlap(candidate_skills, self.REQUIRED, self.PREFERRED), 10.0)

    def test_missing_critical_skill_costs_more_than_missing_minor_skill(self):
        missing_critical = ['Django', 'PostgreSQL', 'Docker', 'Redis']  # missing Python (weight 5)
        missing_minor = ['Python', 'Django', 'PostgreSQL', 'Docker']  # missing Redis (weight 1)

        score_missing_critical = deterministic_skill_overlap(missing_critical, self.REQUIRED, self.PREFERRED)
        score_missing_minor = deterministic_skill_overlap(missing_minor, self.REQUIRED, self.PREFERRED)

        self.assertLess(score_missing_critical, score_missing_minor)

    def test_no_overlap_scores_zero(self):
        candidate_skills = ['Java', 'Spring Boot', 'Oracle DB']
        self.assertEqual(deterministic_skill_overlap(candidate_skills, self.REQUIRED, self.PREFERRED), 0.0)

    def test_no_requirements_falls_back_to_neutral_score(self):
        self.assertEqual(deterministic_skill_overlap(['Python'], [], []), 5.0)

    @override_settings(LLM_SCORE_WEIGHT=0.6, DETERMINISTIC_SCORE_WEIGHT=0.4)
    def test_combine_scores_applies_configured_weights(self):
        self.assertEqual(combine_scores(llm_semantic_score=8.0, deterministic_score=5.0), 6.8)
