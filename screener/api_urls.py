from django.urls import path

from . import api

urlpatterns = [
    path('job-descriptions/', api.job_description_list_create, name='api-jd-list-create'),
    path('resumes/upload/', api.resume_upload, name='api-resume-upload'),
    path('match/', api.run_match, name='api-run-match'),
    path(
        'job-descriptions/<int:job_description_id>/shortlist/',
        api.shortlist,
        name='api-shortlist',
    ),
    path('resumes/<int:resume_id>/check-other-roles/', api.check_other_roles, name='api-check-other-roles'),
]
