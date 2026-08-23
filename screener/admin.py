from django.contrib import admin

from .models import JobDescription, Match, Resume


@admin.register(JobDescription)
class JobDescriptionAdmin(admin.ModelAdmin):
    list_display = ('title', 'created_at')
    readonly_fields = ('created_at',)


@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = ('candidate_name', 'file', 'created_at')
    readonly_fields = ('created_at',)


@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ('resume', 'job_description', 'final_score', 'created_at')
    list_filter = ('job_description',)
    readonly_fields = ('created_at', 'updated_at')
