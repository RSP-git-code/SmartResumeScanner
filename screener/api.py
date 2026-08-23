"""DRF JSON API. Every view here is a thin wrapper around
screener/text_extraction.py and screener/llm/pipeline.py -- no business
logic lives here, so the same behavior is exercised whether a caller uses
the API or the HTML dashboard in views.py.
"""

from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import text_extraction
from .llm import pipeline
from .models import JobDescription, Resume
from .serializers import JobDescriptionSerializer, MatchSerializer, ResumeSerializer


@api_view(['GET', 'POST'])
def job_description_list_create(request):
    if request.method == 'GET':
        jds = JobDescription.objects.order_by('-created_at')
        return Response(JobDescriptionSerializer(jds, many=True).data)

    title = (request.data.get('title') or '').strip()
    text = (request.data.get('text') or '').strip()
    file_obj = request.FILES.get('file')

    if not title:
        return Response({'detail': 'title is required'}, status=status.HTTP_400_BAD_REQUEST)
    if not text and not file_obj:
        return Response({'detail': 'text or file is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        raw_text = text or text_extraction.extract_text(file_obj, filename=file_obj.name)
    except ValueError as exc:
        return Response({'detail': str(exc)}, status=status.HTTP_400_BAD_REQUEST)

    jd = JobDescription.objects.create(title=title, raw_text=raw_text)
    pipeline.process_job_description(jd)
    return Response(JobDescriptionSerializer(jd).data, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def resume_upload(request):
    files = request.FILES.getlist('files')
    if not files:
        return Response({'detail': 'files is required'}, status=status.HTTP_400_BAD_REQUEST)

    created = []
    for f in files:
        try:
            raw_text = text_extraction.extract_text(f, filename=f.name)
        except ValueError as exc:
            return Response({'detail': f'{f.name}: {exc}'}, status=status.HTTP_400_BAD_REQUEST)
        resume = Resume.objects.create(file=f, raw_text=raw_text)
        pipeline.process_resume(resume)
        created.append(resume)

    return Response(ResumeSerializer(created, many=True).data, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def run_match(request):
    job_description_id = request.data.get('job_description_id')
    resume_ids = request.data.get('resume_ids') or []

    if not job_description_id:
        return Response({'detail': 'job_description_id is required'}, status=status.HTTP_400_BAD_REQUEST)

    jd = get_object_or_404(JobDescription, pk=job_description_id)
    resumes = Resume.objects.filter(pk__in=resume_ids) if resume_ids else Resume.objects.all()

    matches = [pipeline.run_match(resume, jd) for resume in resumes]
    return Response(MatchSerializer(matches, many=True).data)


@api_view(['GET'])
def shortlist(request, job_description_id):
    jd = get_object_or_404(JobDescription, pk=job_description_id)
    matches = jd.matches.select_related('resume').order_by('-final_score')
    return Response(MatchSerializer(matches, many=True).data)


@api_view(['POST'])
def check_other_roles(request, resume_id):
    """Scores one candidate against every open JD except the one named in
    `exclude_job_description_id`, surfacing roles they may fit better than
    the one they were originally screened against.
    """
    resume = get_object_or_404(Resume, pk=resume_id)
    exclude_id = request.data.get('exclude_job_description_id')

    other_jds = JobDescription.objects.all()
    if exclude_id:
        other_jds = other_jds.exclude(id=exclude_id)

    matches = [pipeline.run_match(resume, jd) for jd in other_jds]
    return Response(MatchSerializer(matches, many=True).data)
