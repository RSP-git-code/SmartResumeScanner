"""Builds the hiring-manager PDF export for a job's shortlist. Consumes the
same context dict `views._build_shortlist_context` produces, so the report
can never show numbers that disagree with the dashboard.
"""

from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BAND_COLORS = {
    'Strong Match': colors.HexColor('#22c55e'),
    'Moderate Match': colors.HexColor('#f59e0b'),
    'Below Cutoff': colors.HexColor('#ef4444'),
}


def build_hiring_manager_report_pdf(jd, matches, ineligible_matches, band_counts,
                                     selected_target=None, selected_cutoff=None, withdrawn_count=0) -> bytes:
    """Renders the PDF and returns it as bytes.

    `matches` are the eligible, ranked candidates (as annotated by
    `views._build_shortlist_context`: qualification_band, final_score_pct,
    skills/experience/education pct, gap_summary, in_interview_pool where a
    cutoff is selected). `ineligible_matches` are excluded candidates with
    `eligibility_details`.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'], fontSize=18, spaceAfter=4)
    subtitle_style = ParagraphStyle('ReportSubtitle', parent=styles['Normal'], textColor=colors.grey, spaceAfter=16)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'], spaceBefore=14, spaceAfter=8)
    body = styles['BodyText']
    small = ParagraphStyle('Small', parent=styles['BodyText'], fontSize=8, leading=10)

    story = []
    story.append(Paragraph('Candidate Shortlist Report', title_style))
    story.append(Paragraph(
        f'{jd.title} &mdash; generated {datetime.now().strftime("%d %b %Y, %H:%M")}',
        subtitle_style,
    ))

    # Executive summary
    story.append(Paragraph('Executive Summary', h2))
    summary_lines = [
        f'{len(matches)} eligible candidate(s) scored '
        f'&mdash; {band_counts.get("strong", 0)} Strong Match, '
        f'{band_counts.get("moderate", 0)} Moderate Match, '
        f'{band_counts.get("below_cutoff", 0)} Below Cutoff.',
        f'{len(ineligible_matches)} candidate(s) excluded on hard eligibility criteria '
        '(see appendix).',
    ]
    if withdrawn_count:
        summary_lines.append(f'{withdrawn_count} candidate(s) withdrawn from consideration, not shown below.')
    if selected_target and selected_cutoff is not None:
        pool_size = sum(1 for m in matches if getattr(m, 'in_interview_pool', False))
        summary_lines.append(
            f'Interview cutoff selected for a target of {selected_target}: score &ge; '
            f'{selected_cutoff}/10 puts {pool_size} candidate(s) in the pool.'
        )
    for line in summary_lines:
        story.append(Paragraph(line, body))

    # Ranked candidate table
    story.append(Paragraph('Ranked Candidates', h2))
    header = ['#', 'Candidate', 'Overall', 'Band', 'Skills', 'Experience', 'Education']
    if selected_target and selected_cutoff is not None:
        header.append('In Pool')
    table_data = [header]
    for i, m in enumerate(matches, start=1):
        row = [
            str(i),
            m.resume.candidate_name or m.resume.file.name,
            f'{m.final_score_pct}%',
            m.qualification_band,
            f'{round(m.skills_pct)}%',
            f'{round(m.experience_pct)}%',
            f'{round(m.education_pct)}%',
        ]
        if selected_target and selected_cutoff is not None:
            row.append('Yes' if getattr(m, 'in_interview_pool', False) else '')
        table_data.append(row)

    col_widths = None
    ranked_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    for row_idx, m in enumerate(matches, start=1):
        band_color = BAND_COLORS.get(m.qualification_band)
        if band_color:
            table_style.append(('TEXTCOLOR', (3, row_idx), (3, row_idx), band_color))
    ranked_table.setStyle(TableStyle(table_style))
    story.append(ranked_table)

    # Gap summaries for anything short of Strong Match
    gap_rows = [m for m in matches if getattr(m, 'gap_summary', None)]
    if gap_rows:
        story.append(Paragraph('Notes on Moderate / Below-Cutoff Candidates', h2))
        for m in gap_rows:
            name = m.resume.candidate_name or m.resume.file.name
            story.append(KeepTogether([
                Paragraph(f'<b>{name}</b>', body),
                Paragraph(m.gap_summary, small),
                Spacer(1, 4),
            ]))

    # Eligibility appendix
    if ineligible_matches:
        story.append(Paragraph('Appendix: Excluded on Eligibility', h2))
        for m in ineligible_matches:
            name = m.resume.candidate_name or m.resume.file.name
            unmet = [c for c in (m.eligibility_details or []) if not c.get('met')]
            reasons = '; '.join(f"{c.get('criterion')} &mdash; {c.get('evidence')}" for c in unmet) or 'Not specified'
            story.append(KeepTogether([
                Paragraph(f'<b>{name}</b>', body),
                Paragraph(reasons, small),
                Spacer(1, 4),
            ]))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        'Methodology: candidates are scored on Skills, Experience, and Education against the '
        'job\'s extracted requirements using RAG-grounded LLM scoring blended with a '
        'deterministic skill-overlap check, then screened against hard eligibility criteria. '
        'Scoring is run on PII-redacted resume text (name, age, gender, and address removed) '
        'to reduce bias.',
        small,
    ))

    doc.build(story)
    return buffer.getvalue()
