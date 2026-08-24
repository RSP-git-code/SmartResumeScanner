from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('resumes/', views.resume_pool, name='resume-pool'),
    path('resumes/<int:resume_id>/retry-extraction/', views.retry_resume_extraction, name='retry-resume-extraction'),
    path('jobs/<int:job_description_id>/', views.job_detail, name='job-detail'),
    path('jobs/<int:job_description_id>/upload-resumes/', views.upload_resumes, name='upload-resumes'),
    path('jobs/<int:job_description_id>/run-matching/', views.run_matching, name='run-matching'),
    path('jobs/<int:job_description_id>/export-report/', views.export_report, name='export-report'),
    path('jobs/<int:job_description_id>/status/', views.update_job_status, name='update-job-status'),
    path(
        'jobs/<int:job_description_id>/resumes/<int:resume_id>/withdraw/',
        views.withdraw_candidate,
        name='withdraw-candidate',
    ),
    path(
        'jobs/<int:job_description_id>/retry-extraction/',
        views.retry_jd_extraction,
        name='retry-jd-extraction',
    ),
    path(
        'jobs/<int:job_description_id>/resumes/<int:resume_id>/check-other-roles/',
        views.check_other_roles,
        name='check-other-roles',
    ),
]
