"""Builds the hiring-manager PDF export for a job's shortlist. Consumes the
same context dict `views._build_shortlist_context` produces, so the report
can never show numbers that disagree with the dashboard.
"""

from datetime import datetime
from io import BytesIO

from reportlab.graphics.shapes import Drawing, Rect, String
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

INK = colors.HexColor('#1e293b')
MUTED = colors.HexColor('#475569')
TRACK = colors.HexColor('#e2e8f0')


def _horizontal_bar_chart(rows, denom, width=460, label_width=190, row_height=22, bar_height=11):
    """Renders (label, value, color) rows as a horizontal bar chart against
    `denom` (each bar's length is value/denom). Returns a reportlab Drawing,
    which is itself a Flowable and can go straight into a platypus story.
    Hand-rolled with plain Rect/String rather than reportlab's chart classes
    -- simpler to keep visually consistent with the dashboard's own
    Bootstrap progress-bar chart for the same numbers.
    """
    height = row_height * len(rows)
    bar_area_width = width - label_width - 55
    d = Drawing(width, height)
    for i, (label, value, color) in enumerate(rows):
        y = height - (i + 1) * row_height + (row_height - bar_height) / 2
        pct = (value / denom * 100) if denom else 0
        bar_width = bar_area_width * min(pct, 100) / 100
        d.add(String(0, y + 2, label, fontSize=8, fillColor=INK))
        d.add(Rect(label_width, y, bar_area_width, bar_height, fillColor=TRACK, strokeColor=None))
        if bar_width > 0:
            d.add(Rect(label_width, y, bar_width, bar_height, fillColor=color, strokeColor=None))
        d.add(String(label_width + bar_area_width + 6, y + 2, f'{value} ({pct:.0f}%)', fontSize=8, fillColor=MUTED))
    return d


def _distribution_chart(band_counts, total):
    rows = [
        ('Strong Match (≥80%)', band_counts.get('strong', 0), BAND_COLORS['Strong Match']),
        ('Moderate Match (50-79%)', band_counts.get('moderate', 0), BAND_COLORS['Moderate Match']),
        ('Below Cutoff (<50%)', band_counts.get('below_cutoff', 0), BAND_COLORS['Below Cutoff']),
    ]
    return _horizontal_bar_chart(rows, total)


def _skill_coverage_chart(skill_matrix, total, max_skills=8):
    """One bar per required skill: what fraction of eligible candidates have
    at least Partial evidence for it (not just Weak/None) -- the same
    Strong/Partial/Weak/None grading behind the dashboard's skill matrix
    (see scoring.py), collapsed to a single coverage number per skill so a
    hiring manager can see which requirements are actually well-covered by
    the candidate pool without reading the full per-candidate matrix.
    Limited to the highest-weighted skills so the chart stays compact.
    """
    if not skill_matrix or not total:
        return None
    ranked = sorted(skill_matrix, key=lambda e: e.get('weight') or 0, reverse=True)[:max_skills]
    rows = [
        (entry.get('skill', ''), sum(1 for lvl in entry.get('evidence', []) if lvl in ('Strong', 'Partial')),
         colors.HexColor('#4f46e5'))
        for entry in ranked
    ]
    return _horizontal_bar_chart(rows, total)


def build_hiring_manager_report_pdf(jd, matches, ineligible_matches, band_counts,
                                     selected_target=None, selected_cutoff=None, withdrawn_count=0,
                                     skill_matrix=None) -> bytes:
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

    story.append(Spacer(1, 8))
    story.append(_distribution_chart(band_counts, len(matches)))

    coverage_chart = _skill_coverage_chart(skill_matrix, len(matches))
    if coverage_chart:
        story.append(Paragraph('Skill Coverage (highest-weighted requirements)', h2))
        story.append(Paragraph(
            'Share of eligible candidates with at least Partial evidence for each skill '
            '(a self-declared buzzword with no supporting detail grades as Weak, not covered).',
            small,
        ))
        story.append(Spacer(1, 4))
        story.append(coverage_chart)

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
