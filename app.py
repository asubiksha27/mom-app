import os, re, tempfile
from datetime import datetime
from collections import defaultdict

import streamlit as st
import whisper
import spacy
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# ---------------- CONFIG ----------------
MODEL_SIZE = "small"
TIMELINE_SEGMENTS = 6

# ---------------- LOAD NLP ----------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

TASK_REGEX = r'(?:(?:To |Assign(ed to )?)?([A-Z][a-zA-Z]{1,30}))?.*?\b(submit|prepare|share|complete|send|deliver|provide|finalize)\b.*?\b(on|by)\b\s+([0-9]{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]+)[-/ ]\d{2,4}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})'

# ---------------- FUNCTIONS ----------------
def transcribe_and_translate(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)
    detected_lang = result.get("language", None)
    segments = result.get("segments", [])
    text = result.get("text", "").strip()

    # Auto-translate if not English
    if detected_lang and detected_lang != "en":
        translated = model.transcribe(audio_path, task="translate", verbose=False)
        segments = translated.get("segments", segments)
        text = translated.get("text", text)
        detected_lang = "en"

    return segments, detected_lang, text


def extract_action_items(segments):
    items = []
    for seg in segments:
        text = seg.get("text","").strip()
        if not text: continue
        m = re.search(TASK_REGEX, text, re.I)
        if m:
            assignee = m.group(2) or "Unassigned"
            deadline = m.group(5) or ""
            items.append((assignee, text, deadline))
    return items


def build_timeline(segments, max_rows=6):
    timeline = []
    for seg in segments[:max_rows]:
        start = seg.get("start",0.0)
        end = seg.get("end", start)
        text = seg.get("text","").replace("\n"," ").strip()
        topic = (text.split(".")[0])[:80] if text else ""
        timeline.append({"start":start, "end":end, "topic": topic})
    return timeline


def generate_summary(full_text):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        out = summarizer(full_text, max_length=150, min_length=50, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return full_text


def write_pdf(path, meeting_date, meeting_time, timeline, summary, action_items, dialogue):
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=10*mm, leftMargin=10*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall', parent=styles['Normal'], fontSize=9, leading=11, alignment=0))
    story = []

    story.append(Paragraph("<b>Minutes of Meeting (MoM)</b>", styles['Heading2']))
    story.append(Spacer(1,6))

    story.append(Paragraph(f"<b>Date:</b> {meeting_date}  |  <b>Time:</b> {meeting_time}", styles['BodySmall']))
    story.append(Spacer(1,6))

    # Timeline
    if timeline:
        story.append(Paragraph("<b>Timeline</b>", styles['BodySmall']))
        table_data = [["Start","End","Topic"]]
        for row in timeline:
            def fmt(s): m = int(s//60); sec = int(s%60); return f"{m:02d}:{sec:02d}"
            table_data.append([fmt(row["start"]), fmt(row["end"]), row["topic"]])
        table = Table(table_data, colWidths=[50,50,300])
        table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.35,colors.black)]))
        story.append(table)
        story.append(Spacer(1,6))

    # Summary
    story.append(Paragraph("<b>Summary</b>", styles['BodySmall']))
    story.append(Paragraph(summary, styles['BodySmall']))
    story.append(Spacer(1,6))

    # Action items
    if action_items:
        story.append(Paragraph("<b>Action Items</b>", styles['BodySmall']))
        table_data = [["Assigned To","Task","Deadline"]] + list(action_items)
        table = Table(table_data, colWidths=[100,250,100])
        table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.35,colors.black)]))
        story.append(table)
        story.append(Spacer(1,6))

    # Dialogue
    story.append(Paragraph("<b>Speaker Dialogue</b>", styles['BodySmall']))
    for speaker, lines in dialogue.items():
        story.append(Paragraph(f"<b>{speaker}</b>", styles['BodySmall']))
        for l in lines:
            story.append(Paragraph(f"• {l}", styles['BodySmall']))

    doc.build(story)


# ---------------- STREAMLIT APP ----------------
st.title("🎙️ AI-Powered MoM Generator")

uploaded_file = st.file_uploader("Upload meeting audio (.mp3 / .wav)", type=["mp3","wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.info("⏳ Processing audio... please wait")

    segments, lang, full_text = transcribe_and_translate(audio_path, model_size=MODEL_SIZE)

    timeline = build_timeline(segments, max_rows=TIMELINE_SEGMENTS)
    action_items = extract_action_items(segments)
    summary = generate_summary(full_text)

    # Simple dialogue grouping
    dialogue = defaultdict(list)
    for seg in segments:
        dialogue["Speaker"].append(seg.get("text",""))

    st.subheader("📌 Summary")
    st.write(summary)

    st.subheader("✅ Action Items")
    if action_items:
        st.table(action_items)
    else:
        st.write("No clear action items detected.")

    # Generate PDF
    pdf_path = "MoM_Output.pdf"
    write_pdf(pdf_path, datetime.now().strftime("%d-%b-%Y"), "10:00 AM - 11:00 AM", timeline, summary, action_items, dialogue)

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download MoM PDF", f, file_name="MoM.pdf", mime="application/pdf")
