from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('jobs/<int:job_description_id>/', views.job_detail, name='job-detail'),
    path('jobs/<int:job_description_id>/upload-resumes/', views.upload_resumes, name='upload-resumes'),
    path('jobs/<int:job_description_id>/run-matching/', views.run_matching, name='run-matching'),
    path(
        'jobs/<int:job_description_id>/resumes/<int:resume_id>/check-other-roles/',
        views.check_other_roles,
        name='check-other-roles',
    ),
]
