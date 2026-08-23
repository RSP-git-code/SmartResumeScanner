from rest_framework import serializers

from .models import JobDescription, Match, Resume


class JobDescriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobDescription
        fields = ['id', 'title', 'raw_text', 'extracted_requirements', 'created_at']
        read_only_fields = ['extracted_requirements', 'created_at']


class ResumeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resume
        fields = [
            'id', 'candidate_name', 'file',
            'extracted_skills', 'extracted_experience', 'extracted_education',
            'created_at',
        ]
        read_only_fields = [
            'candidate_name', 'extracted_skills', 'extracted_experience',
            'extracted_education', 'created_at',
        ]


class MatchSerializer(serializers.ModelSerializer):
    resume = ResumeSerializer(read_only=True)

    class Meta:
        model = Match
        fields = [
            'id', 'resume', 'job_description',
            'skills_subscore', 'experience_subscore', 'education_subscore',
            'skills_justification', 'experience_justification', 'education_justification',
            'evidence_quotes', 'deterministic_score', 'llm_semantic_score', 'final_score',
            'interview_questions', 'created_at',
        ]
        read_only_fields = fields
