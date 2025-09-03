import os, re, json, tempfile
from datetime import datetime
from collections import defaultdict

import streamlit as st
import whisper
import spacy
from transformers import pipeline
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# ---------------- CONFIG ----------------
MODEL_SIZE = "small"
TIMELINE_SEGMENTS = 6

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model(MODEL_SIZE)

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

whisper_model = load_whisper_model()
nlp = load_spacy_model()
summarizer = load_summarizer()

# ---------------- HELPERS ----------------
TASK_REGEX = r'(?:(?:To |Assign(ed to )?)?([A-Z][a-zA-Z]{1,30}))?.*?\b(submit|prepare|share|complete|send|deliver|provide|finalize)\b.*?\b(on|by)\b\s+([0-9]{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]+)[-/ ]\d{2,4}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})'

def transcribe_and_translate(audio_path):
    result = whisper_model.transcribe(audio_path, verbose=False)
    detected_lang = result.get("language", None)
    segments = result.get("segments", [])
    text = result.get("text", "").strip()

    # Translate if not English
    if detected_lang and detected_lang != "en":
        translated = whisper_model.transcribe(audio_path, task="translate", verbose=False)
        segments = translated.get("segments", segments)
        text = translated.get("text", text)
        detected_lang = "en"

    # Normalize
    uniform = []
    for s in segments:
        uniform.append({"start": s.get("start",0.0), "end": s.get("end",0.0), "text": s.get("text","")})
    return uniform, detected_lang, text

def build_timeline(segments, max_rows=6):
    timeline = []
    for seg in segments[:max_rows]:
        start = seg.get("start",0.0)
        end = seg.get("end", start)
        text = seg.get("text","").replace("\n"," ").strip()
        topic = (text.split(".")[0])[:80] if text else ""
        timeline.append({"start":start, "end":end, "topic": topic})
    return timeline

def extract_action_items(segments):
    action_items = []
    for seg in segments:
        text = seg.get("text","").strip()
        if not text: continue
        m = re.search(TASK_REGEX, text, re.I)
        if m:
            assignee = m.group(2) or "Unassigned"
            deadline = m.group(5) or ""
            action_items.append((assignee, text, deadline))
    return action_items

def generate_summary(full_text):
    try:
        out = summarizer(full_text, max_length=400, min_length=100, do_sample=False)
        return out[0]["summary_text"]
    except:
        return full_text

def write_mom_pdf(path, meeting_date, meeting_time, location, participants, timeline, summary, dialogue, action_items):
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=10*mm, leftMargin=10*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall', parent=styles['Normal'], fontSize=9, leading=11))

    story = []

    # Header
    story.append(Paragraph("<b>Minutes of Meeting (MoM)</b>", styles['Heading2']))
    story.append(Spacer(1,6))
    meta = f"<b>Meeting Date:</b> {meeting_date} &nbsp;&nbsp; <b>Time:</b> {meeting_time} &nbsp;&nbsp; <b>Location:</b> {location}"
    story.append(Paragraph(meta, styles['BodySmall']))
    if participants:
        story.append(Paragraph("<b>Participants:</b> " + ", ".join(participants), styles['BodySmall']))
    story.append(Spacer(1,6))

    # Timeline
    if timeline:
        story.append(Paragraph("<b>Meeting Timeline</b>", styles['BodySmall']))
        table_data = [["Start","End","Topic"]]
        for row in timeline:
            def fmt(s): m = int(s//60); sec = int(s%60); return f"{m:02d}:{sec:02d}"
            table_data.append([fmt(row["start"]), fmt(row["end"]), row["topic"]])
        table = Table(table_data, colWidths=[45,45,350])
        table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.35,colors.black)]))
        story.append(table)
        story.append(Spacer(1,6))

    # Summary
    story.append(Paragraph("<b>Meeting Summary</b>", styles['BodySmall']))
    story.append(Paragraph(summary, styles['BodySmall']))
    story.append(Spacer(1,6))

    # Speaker Dialogue
    story.append(Paragraph("<b>Speaker-wise Dialogue</b>", styles['BodySmall']))
    for speaker, lines in dialogue.items():
        story.append(Paragraph(f"<b>{speaker}</b>", styles['BodySmall']))
        for l in lines:
            story.append(Paragraph(f"• {l}", styles['BodySmall']))
    story.append(Spacer(1,6))

    # Action Items
    if action_items:
        story.append(Paragraph("<b>Action Items</b>", styles['BodySmall']))
        table_data = [["Assigned To","Task","Deadline"]] + action_items
        table = Table(table_data, colWidths=[100,310,60])
        table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.35,colors.black)]))
        story.append(table)

    doc.build(story)

# ---------------- STREAMLIT APP ----------------
st.title("📑 AI Minutes of Meeting (MoM) Generator")

uploaded_file = st.file_uploader("Upload a meeting audio", type=["mp3","wav"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    with st.spinner("Processing audio..."):
        segments, lang, full_text = transcribe_and_translate(audio_path)
        timeline = build_timeline(segments, max_rows=TIMELINE_SEGMENTS)
        action_items = extract_action_items(segments)

        # Simple participant + dialogue extraction
        participants = []
        dialogue = defaultdict(list)
        for seg in segments:
            text = seg["text"].strip()
            speaker = "Unknown"
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    speaker = ent.text
                    break
            participants.append(speaker)
            dialogue[speaker].append(text)

        participants = list(set(participants))
        summary = generate_summary(full_text)

        meeting_date = datetime.now().strftime("%d-%b-%Y")
        meeting_time = "10:00 AM - 11:00 AM"
        location = "Virtual"

        pdf_path = "MoM_Output.pdf"
        write_mom_pdf(pdf_path, meeting_date, meeting_time, location, participants, timeline, summary, dialogue, action_items)

    st.success("✅ MoM generated successfully!")
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download MoM PDF", f, file_name="meeting_minutes.pdf", mime="application/pdf")
