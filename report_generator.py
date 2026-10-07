import io
import datetime
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(leaf_image: Image.Image, display_name: str, category: str, confidence: float, severity: str, info: dict, top3: list):
    """
    Generates a professional 1-page PDF diagnostic report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#059669'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        textColor=colors.HexColor('#334155'),
        leading=13
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#94A3B8'),
        alignment=1,
        spaceBefore=12
    )

    story = []

    # Title & Header
    story.append(Paragraph("🌿 FloraScan AI · Clinical Plant Pathology Report", title_style))
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S")
    story.append(Paragraph(f"Generated on {now_str} | Deep Learning CNN Diagnosis (VGG16 Model)", subtitle_style))

    # Prepare Leaf Thumbnail
    img_buffer = io.BytesIO()
    # Save a resized copy for PDF
    thumb = leaf_image.copy()
    thumb.thumbnail((180, 180))
    thumb.save(img_buffer, format="JPEG", quality=85)
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=130, height=130)

    # Summary table data
    summary_data = [
        [Paragraph("<b>Diagnosed Condition:</b>", body_style), Paragraph(f"<b><font color='#059669' size='11'>{display_name}</font></b>", body_style)],
        [Paragraph("<b>Classification Category:</b>", body_style), Paragraph(f"{category}", body_style)],
        [Paragraph("<b>Model Confidence:</b>", body_style), Paragraph(f"<b>{confidence:.2f}%</b>", body_style)],
        [Paragraph("<b>Disease Severity:</b>", body_style), Paragraph(f"{severity}", body_style)],
        [Paragraph("<b>Diagnostic Status:</b>", body_style), Paragraph("Validated (Real-Time Inference)", body_style)],
    ]

    summary_table = Table(summary_data, colWidths=[130, 200])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))

    # Main header layout table (Thumbnail on left, Summary table on right)
    header_table = Table([[rl_img, summary_table]], colWidths=[150, 390])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (0,0), (0,0), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # Top 3 Candidates
    story.append(Paragraph("Probabilistic Breakdown (Top 3 Candidates)", section_heading))
    top3_rows = [[Paragraph("<b>Rank</b>", body_style), Paragraph("<b>Candidate Condition</b>", body_style), Paragraph("<b>Softmax Probability</b>", body_style)]]
    for idx, (name, prob) in enumerate(top3, start=1):
        top3_rows.append([
            Paragraph(f"#{idx}", body_style),
            Paragraph(f"{name}", body_style),
            Paragraph(f"{prob:.2f}%", body_style)
        ])
    t3_table = Table(top3_rows, colWidths=[50, 340, 150])
    t3_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#E2E8F0')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t3_table)
    story.append(Spacer(1, 10))

    # Clinical Description
    story.append(Paragraph("Clinical Pathology Description", section_heading))
    desc = info.get("description", "No detailed description provided.")
    story.append(Paragraph(desc, body_style))
    story.append(Spacer(1, 6))

    # Prescribed Treatment
    story.append(Paragraph("Recommended Treatment Protocol", section_heading))
    treatment = info.get("treatment", "No treatment available.")
    story.append(Paragraph(treatment, body_style))
    story.append(Spacer(1, 6))

    # Prevention Measures
    story.append(Paragraph("Agronomic Prevention & Cultural Practices", section_heading))
    prevention = info.get("prevention", "No prevention guidelines available.")
    story.append(Paragraph(prevention, body_style))
    story.append(Spacer(1, 12))

    # Disclaimer
    story.append(Paragraph(
        "Disclaimer: This diagnostic report is generated autonomously by an AI Computer Vision Model (VGG16 architecture). "
        "Intended for advisory and educational purposes. Always consult local certified agricultural extension personnel before extensive chemical application.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
