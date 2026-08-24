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

        # The qualification-distribution summary always shows a "Not
        # Eligible" *count* card (0 here), so check the actual ineligible
        # section/accordion specifically rather than that substring.
        self.assertEqual(response.context['ineligible_matches'], [])
        self.assertNotContains(response, 'id="ineligibleAccordion"')


class QualificationBandAndInterviewPlannerTests(TestCase):
    """Qualification bands, band-count summary, and the interview-target
    cutoff planner (screener/views.py:job_detail) -- all pure computation
    over already-scored Match fixtures, no LLM call needed.
    """

    def _make_matches(self, jd):
        scores = [9.0, 8.5, 7.0, 6.0, 3.0]
        for i, score in enumerate(scores):
            resume = Resume.objects.create(file=f'resumes/{i}.pdf', raw_text='...', candidate_name=f'Candidate {i}')
            Match.objects.create(resume=resume, job_description=jd, final_score=score, is_eligible=True)
        return scores

    def test_band_counts_and_labels(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        self._make_matches(jd)

        response = self.client.get(f'/jobs/{jd.id}/')

        # 9.0, 8.5 -> strong (>=8); 7.0, 6.0 -> moderate (5-7.9); 3.0 -> below cutoff
        self.assertEqual(response.context['band_counts'], {'strong': 2, 'moderate': 2, 'below_cutoff': 1})
        self.assertContains(response, 'Strong Match')
        self.assertContains(response, 'Moderate Match')
        self.assertContains(response, 'Below Cutoff')

    def test_gap_summary_only_shown_below_strong_match(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        self._make_matches(jd)

        response = self.client.get(f'/jobs/{jd.id}/')
        matches = {m.final_score: m for m in response.context['matches']}

        self.assertFalse(hasattr(matches[9.0], 'gap_summary'))
        self.assertTrue(hasattr(matches[6.0], 'gap_summary'))
        self.assertIn('Main gap', matches[6.0].gap_summary)

    def test_interview_target_marks_pool_membership(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        self._make_matches(jd)  # 5 candidates: 9.0, 8.5, 7.0, 6.0, 3.0

        # With only 5 candidates, target=5 ("All") is the only valid option (5/10/15/20 are all >= total).
        response = self.client.get(f'/jobs/{jd.id}/', {'interview_target': 5})

        self.assertEqual(response.context['selected_cutoff'], 3.0)
        pool = {m.final_score: m.in_interview_pool for m in response.context['matches']}
        self.assertTrue(all(pool.values()))  # cutoff is the lowest score, so everyone qualifies


class SkillEvidenceMatrixTests(TestCase):
    """Skill Ranking Matrix now shows per-skill evidence strength
    (Strong/Partial/Weak/None) from Match.skill_evidence rather than a
    binary has_skill check (screener/views.py:_skill_evidence_level).
    """

    def test_matrix_uses_graded_evidence_level(self):
        jd = JobDescription.objects.create(
            title='Backend Engineer',
            raw_text='...',
            extracted_requirements={
                'required_skills': [{'skill': 'Python', 'weight': 5}, {'skill': 'Kubernetes', 'weight': 3}],
            },
        )
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Candidate')
        Match.objects.create(
            resume=resume,
            job_description=jd,
            final_score=8.0,
            is_eligible=True,
            skill_evidence=[
                {'skill': 'Python', 'evidence_level': 'Strong', 'evidence_quote': 'Built a Django service...'},
                {'skill': 'Kubernetes', 'evidence_level': 'Weak', 'evidence_quote': None},
            ],
        )

        response = self.client.get(f'/jobs/{jd.id}/')

        matrix = {row['skill']: row['evidence'] for row in response.context['skill_matrix']}
        self.assertEqual(matrix['Python'], ['Strong'])
        self.assertEqual(matrix['Kubernetes'], ['Weak'])
        self.assertContains(response, 'evidence-strong')
        self.assertContains(response, 'evidence-weak')

    def test_skill_missing_from_evidence_defaults_to_none(self):
        jd = JobDescription.objects.create(
            title='Backend Engineer',
            raw_text='...',
            extracted_requirements={'required_skills': [{'skill': 'Go', 'weight': 4}]},
        )
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Candidate')
        Match.objects.create(resume=resume, job_description=jd, final_score=8.0, is_eligible=True)

        response = self.client.get(f'/jobs/{jd.id}/')

        [row] = response.context['skill_matrix']
        self.assertEqual(row['evidence'], ['None'])


class ExportReportTests(TestCase):
    """PDF export (screener/reports.py, views.export_report) reuses the same
    context helper as job_detail, so it just needs to return a valid PDF
    response -- content accuracy is covered by the shared context already
    being tested in QualificationBandAndInterviewPlannerTests etc.
    """

    def test_export_returns_pdf(self):
        jd = JobDescription.objects.create(
            title='Backend Engineer',
            raw_text='...',
            extracted_requirements={'required_skills': [{'skill': 'Python', 'weight': 5}]},
        )
        eligible = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Eligible Candidate')
        ineligible = Resume.objects.create(file='resumes/b.pdf', raw_text='...', candidate_name='Ineligible Candidate')
        Match.objects.create(
            resume=eligible, job_description=jd, final_score=8.0, is_eligible=True,
            skill_evidence=[{'skill': 'Python', 'evidence_level': 'Strong', 'evidence_quote': None}],
        )
        Match.objects.create(
            resume=ineligible, job_description=jd, final_score=6.0, is_eligible=False,
            eligibility_details=[{'criterion': 'Min 65% aggregate', 'met': False, 'evidence': 'not stated'}],
        )

        response = self.client.get(f'/jobs/{jd.id}/export-report/')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertTrue(response.content.startswith(b'%PDF'))

    def test_export_works_and_button_shown_with_zero_eligible_matches(self):
        """Regression: the "Download Report" button used to be gated on
        `matches` (eligible only), so a job where every candidate failed
        eligibility silently lost the export button -- exactly the report
        that's most useful there (showing why everyone was rejected).
        """
        jd = JobDescription.objects.create(title='Investment Risk Analyst', raw_text='...')
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Om Prakash')
        Match.objects.create(
            resume=resume, job_description=jd, final_score=1.86, is_eligible=False,
            eligibility_details=[
                {'criterion': 'Minimum 4 to 6 years relevant work experience', 'met': False,
                 'evidence': 'No full-time experience.'},
            ],
        )

        detail = self.client.get(f'/jobs/{jd.id}/')
        self.assertContains(detail, 'Download Report')

        response = self.client.get(f'/jobs/{jd.id}/export-report/')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b'%PDF'))

    def test_export_respects_interview_target_query_param(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        for i, score in enumerate([9.0, 8.5, 7.0, 6.0, 3.0]):
            resume = Resume.objects.create(file=f'resumes/{i}.pdf', raw_text='...', candidate_name=f'Candidate {i}')
            Match.objects.create(resume=resume, job_description=jd, final_score=score, is_eligible=True)

        response = self.client.get(f'/jobs/{jd.id}/export-report/', {'interview_target': 5})

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b'%PDF'))


class WithdrawCandidateTests(TestCase):
    """A withdrawn candidate is excluded from the active shortlist/eligible
    split but never deleted, and can be restored (screener/views.py:
    withdraw_candidate, _build_shortlist_context).
    """

    def test_withdraw_removes_from_shortlist_but_keeps_record(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Candidate')
        match = Match.objects.create(resume=resume, job_description=jd, final_score=8.0, is_eligible=True)

        response = self.client.post(f'/jobs/{jd.id}/resumes/{resume.id}/withdraw/')

        self.assertRedirects(response, f'/jobs/{jd.id}/')
        match.refresh_from_db()
        self.assertTrue(match.withdrawn)

        detail = self.client.get(f'/jobs/{jd.id}/')
        self.assertEqual(detail.context['matches'], [])
        self.assertEqual([m.resume for m in detail.context['withdrawn_matches']], [resume])

    def test_withdraw_toggles_back_to_restore(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Candidate')
        match = Match.objects.create(
            resume=resume, job_description=jd, final_score=8.0, is_eligible=True, withdrawn=True
        )

        self.client.post(f'/jobs/{jd.id}/resumes/{resume.id}/withdraw/')

        match.refresh_from_db()
        self.assertFalse(match.withdrawn)
        detail = self.client.get(f'/jobs/{jd.id}/')
        self.assertEqual([m.resume for m in detail.context['matches']], [resume])
        self.assertEqual(detail.context['withdrawn_matches'], [])


class JobStatusTests(TestCase):
    """Marking a role Filled/Closed stops it from being offered as a
    cross-role recommendation target, but leaves existing Match records and
    its own shortlist untouched (screener/views.py: update_job_status,
    check_other_roles, _build_shortlist_context).
    """

    def test_update_job_status(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')

        response = self.client.post(f'/jobs/{jd.id}/status/', {'status': 'filled'})

        self.assertRedirects(response, f'/jobs/{jd.id}/')
        jd.refresh_from_db()
        self.assertEqual(jd.status, JobDescription.STATUS_FILLED)

    def test_invalid_status_rejected(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')

        self.client.post(f'/jobs/{jd.id}/status/', {'status': 'bogus'})

        jd.refresh_from_db()
        self.assertEqual(jd.status, JobDescription.STATUS_OPEN)

    @patch('screener.views.pipeline.run_match')
    def test_closed_role_excluded_from_cross_role_check(self, mock_run_match):
        open_role = JobDescription.objects.create(title='DevOps Engineer', raw_text='...')
        closed_role = JobDescription.objects.create(
            title='Data Analyst', raw_text='...', status=JobDescription.STATUS_CLOSED
        )
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Candidate')
        Match.objects.create(resume=resume, job_description=open_role, final_score=6.0, is_eligible=True)

        self.client.post(f'/jobs/{open_role.id}/resumes/{resume.id}/check-other-roles/')

        mock_run_match.assert_not_called()

    def test_filled_role_not_suggested_as_better_fit(self):
        current_role = JobDescription.objects.create(title='DevOps Engineer', raw_text='...')
        filled_role = JobDescription.objects.create(
            title='Data Analyst', raw_text='...', status=JobDescription.STATUS_FILLED
        )
        resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Candidate')
        Match.objects.create(resume=resume, job_description=current_role, final_score=4.0, is_eligible=True)
        Match.objects.create(resume=resume, job_description=filled_role, final_score=9.0, is_eligible=True)

        response = self.client.get(f'/jobs/{current_role.id}/')

        [match] = response.context['matches']
        self.assertIsNone(match.better_fit_elsewhere)


class ResumePoolTests(TestCase):
    """The Resume Pool page (screener/views.py:resume_pool) lists every
    resume across all jobs with which roles it's matched against -- pure
    ORM/view test, no LLM call needed.
    """

    def test_lists_resumes_with_their_matches(self):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')
        matched = Resume.objects.create(file='resumes/a.pdf', raw_text='...', candidate_name='Matched Candidate')
        unmatched = Resume.objects.create(file='resumes/b.pdf', raw_text='...', candidate_name='Unmatched Candidate')
        Match.objects.create(resume=matched, job_description=jd, final_score=8.0, is_eligible=True)

        response = self.client.get('/resumes/')

        self.assertEqual(response.status_code, 200)
        resumes = list(response.context['resumes'])
        self.assertEqual({r.id for r in resumes}, {matched.id, unmatched.id})
        self.assertContains(response, 'Matched Candidate')
        self.assertContains(response, 'Unmatched Candidate')
        self.assertContains(response, 'Backend Engineer')
        self.assertContains(response, 'Not matched against any job yet')

    def test_empty_state(self):
        response = self.client.get('/resumes/')

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'No resumes uploaded yet')


class RunMatchingScopeTests(TestCase):
    """"Run Matching" defaults to only resumes uploaded for THIS job
    (Resume.uploaded_for), not the entire shared pool -- matching every
    unrelated resume from every other job by default was overloading
    Gemini's free tier and blowing through gunicorn's request timeout on
    jobs that had nothing to do with those candidates. Checking the wider
    pool is available via scope=pool, as an explicit opt-in (screener/
    views.py: run_matching, upload_resumes).
    """

    @patch('screener.views.pipeline.run_match')
    def test_default_scope_only_matches_own_uploads(self, mock_run_match):
        job_a = JobDescription.objects.create(title='Job A', raw_text='...')
        job_b = JobDescription.objects.create(title='Job B', raw_text='...')
        own_resume = Resume.objects.create(file='resumes/a.pdf', raw_text='...', uploaded_for=job_b)
        other_resume = Resume.objects.create(file='resumes/b.pdf', raw_text='...', uploaded_for=job_a)

        self.client.post(f'/jobs/{job_b.id}/run-matching/')

        mock_run_match.assert_called_once_with(own_resume, job_b)

    @patch('screener.views.pipeline.run_match')
    def test_pool_scope_matches_every_unmatched_resume(self, mock_run_match):
        job_a = JobDescription.objects.create(title='Job A', raw_text='...')
        job_b = JobDescription.objects.create(title='Job B', raw_text='...')
        Resume.objects.create(file='resumes/a.pdf', raw_text='...', uploaded_for=job_b)
        Resume.objects.create(file='resumes/b.pdf', raw_text='...', uploaded_for=job_a)

        self.client.post(f'/jobs/{job_b.id}/run-matching/', {'scope': 'pool'})

        self.assertEqual(mock_run_match.call_count, 2)

    @patch('screener.views.pipeline.process_resume')
    def test_upload_sets_uploaded_for(self, mock_process_resume):
        jd = JobDescription.objects.create(title='Backend Engineer', raw_text='...')

        self.client.post(
            f'/jobs/{jd.id}/upload-resumes/',
            {'resumes': [SimpleUploadedFile('resume.txt', b'Ananya Verma\nPython, Django')]},
        )

        [resume] = Resume.objects.all()
        self.assertEqual(resume.uploaded_for, jd)

    def test_job_detail_splits_own_vs_pool_unmatched_counts(self):
        job_a = JobDescription.objects.create(title='Job A', raw_text='...')
        job_b = JobDescription.objects.create(title='Job B', raw_text='...')
        Resume.objects.create(file='resumes/a.pdf', raw_text='...', uploaded_for=job_b)
        Resume.objects.create(file='resumes/b.pdf', raw_text='...', uploaded_for=job_a)

        response = self.client.get(f'/jobs/{job_b.id}/')

        self.assertEqual(len(response.context['own_unmatched_resumes']), 1)
        self.assertEqual(len(response.context['pool_unmatched_resumes']), 1)


class ResumeDedupTests(TestCase):
    """Uploading the same resume content twice must not create two Resume
    rows (screener/views.py:upload_resumes). Mocks pipeline.process_resume
    since real extraction needs a live LLM call -- this test only covers the
    content-hash dedup check that runs before that call.
    """

    @patch('screener.views.pipeline.process_resume')
    def test_duplicate_content_is_skipped_once_extraction_succeeded(self, mock_process_resume):
        mock_process_resume.side_effect = lambda resume: setattr(resume, 'redacted_text', '[CANDIDATE]...') or resume.save()
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

    @patch('screener.views.pipeline.process_resume')
    def test_reuploading_content_retries_if_extraction_never_completed(self, mock_process_resume):
        """content_hash is set at Resume creation time, before extraction
        runs -- if extraction then fails (e.g. a transient Gemini error),
        `redacted_text` is left blank. Re-uploading the same content must
        retry extraction on that existing row, not skip it as a duplicate
        forever (screener/views.py:upload_resumes).
        """
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
        self.assertEqual(mock_process_resume.call_count, 2)


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
