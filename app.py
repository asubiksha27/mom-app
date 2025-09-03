import os, re, json, tempfile
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
MODEL_SIZE = "small"          # "tiny","base","small","medium" etc.
TIMELINE_SEGMENTS = 6         # number of timeline rows
# ----------------------------------------

# ---------- Load models ----------
@st.cache_resource
def load_whisper():
    return whisper.load_model(MODEL_SIZE)

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

whisper_model = load_whisper()
nlp = load_spacy()
summarizer = load_summarizer()

# ---------- Transcription ----------
def transcribe_and_translate_if_needed(audio_path):
    result = whisper_model.transcribe(audio_path, verbose=False)
    detected_lang = result.get("language", None)
    segments = result.get("segments", None) or []
    full_text = result.get("text", "").strip()

    if detected_lang and detected_lang != "en":
        try:
            translated = whisper_model.transcribe(audio_path, task="translate", verbose=False)
            segments = translated.get("segments", segments) or segments
            full_text = translated.get("text", full_text).strip()
            detected_lang = "en"
        except Exception:
            pass

    uniform = []
    for s in segments:
        uniform.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", s.get("start", 0.0))),
            "text": (s.get("text") or "").strip()
        })
    if not uniform and full_text:
        uniform = [{"start": 0.0, "end": 0.0, "text": full_text}]
    return uniform, detected_lang, full_text

# ---------- Key-point extraction ----------
TASK_REGEX = r'(?:(?:To |Assign(ed to )?)?([A-Z][a-zA-Z]{1,30}))?.*?\b(submit|prepare|share|complete|send|deliver|provide|finalize)\b.*?\b(on|by)\b\s+([0-9]{1,2}[-/ ](?:Jan|Feb|Mar|...|Dec|[A-Za-z]+)[-/ ]\d{2,4}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})'

def extract_key_points_with_deadlines(segments):
    action_items = []
    for seg in segments:
        text = seg.get("text","").strip()
        if not text: continue
        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            m = re.search(TASK_REGEX, s, re.I)
            if m:
                assignee = (m.group(2) or "Unassigned") if m.groups() else "Unassigned"
                deadline = m.group(5) if m.lastindex and m.lastindex>=5 else ""
                action_items.append((assignee, s, deadline))
    return action_items

# ---------- Build timeline ----------
def build_timeline(segments, max_rows=6):
    timeline = []
    for seg in segments[:max_rows]:
        start = seg.get("start",0.0); end = seg.get("end", start)
        text = seg.get("text","").replace("\n"," ").strip()
        topic = (text.split(".")[0])[:80] if text else ""
        timeline.append({"start":start, "end":end, "topic": topic})
    return timeline

# ---------- PDF writer ----------
def write_mom_pdf(path, meeting_date, meeting_time, location, participants, timeline, summary_text, speaker_dialogue, action_items):
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=10*mm, leftMargin=10*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall', parent=styles['Normal'], fontSize=9, leading=11, alignment=0))
    styles.add(ParagraphStyle(name='HeadingSmall', parent=styles['Heading2'], fontSize=11, leading=12, alignment=1))
    story = []

    story.append(Paragraph("<b>Minutes of Meeting (MoM)</b>", styles['HeadingSmall']))
    story.append(Spacer(1,4))
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
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.grey),
            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("ALIGN",(0,0),(-1,-1),"LEFT"),
            ("GRID",(0,0),(-1,-1),0.35,colors.black),
            ("FONTSIZE",(0,0),(-1,-1),8)
        ]))
        story.append(table)
        story.append(Spacer(1,6))

    # Summary
    story.append(Paragraph("<b>Meeting Summary</b>", styles['BodySmall']))
    story.append(Paragraph(summary_text, styles['BodySmall']))
    story.append(Spacer(1,6))

    # Speaker Dialogue
    story.append(Paragraph("<b>Speaker-wise Dialogue</b>", styles['BodySmall']))
    for speaker, lines in speaker_dialogue.items():
        story.append(Paragraph(f"<b>{speaker}</b>", styles['BodySmall']))
        for l in lines:
            story.append(Paragraph(f"• {l}", styles['BodySmall']))
    story.append(Spacer(1,6))

    # Action Items
    if action_items:
        story.append(Paragraph("<b>Action Items</b>", styles['BodySmall']))
        table_data = [["Assigned To","Task","Deadline"]]
        for a in action_items:
            table_data.append([a[0], a[1], a[2]])
        table = Table(table_data, colWidths=[100,310,60])
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.grey),
            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("ALIGN",(0,0),(-1,-1),"LEFT"),
            ("GRID",(0,0),(-1,-1),0.35,colors.black),
            ("FONTSIZE",(0,0),(-1,-1),8)
        ]))
        story.append(table)

    doc.build(story)

# ---------- Main Streamlit App ----------
st.title("🎙️ AI Minutes of Meeting Generator (Audio Only)")

uploaded_file = st.file_uploader("Upload meeting audio", type=["mp3","wav","m4a"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    with st.spinner("Transcribing..."):
        segments, detected_lang, full_text = transcribe_and_translate_if_needed(audio_path)

    st.subheader("Transcript")
    st.write(full_text[:2000] + ("..." if len(full_text) > 2000 else ""))

    timeline = build_timeline(segments, max_rows=TIMELINE_SEGMENTS)
    action_items = extract_key_points_with_deadlines(segments)

    # Speaker dialogue grouping
    speaker_dialogue = defaultdict(list)
    participants = []
    for seg in segments:
        text = seg["text"]
        assigned = "Unknown"
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                assigned = ent.text
                break
        speaker_dialogue[assigned].append(text)
        if assigned not in participants:
            participants.append(assigned)

    # Generate summary
    with st.spinner("Summarizing..."):
        try:
            out = summarizer(full_text, max_length=500, min_length=120, do_sample=False)
            summary_text = out[0]["summary_text"]
        except:
            summary_text = full_text

    st.subheader("Meeting Summary")
    st.write(summary_text)

    # PDF Export
    pdf_path = os.path.join(tempfile.gettempdir(), "meeting_mom.pdf")
    meeting_date = datetime.now().strftime("%d-%b-%Y")
    meeting_time = "10:00 AM - 11:00 AM"
    location = "Virtual"

    write_mom_pdf(pdf_path, meeting_date, meeting_time, location, participants, timeline, summary_text, speaker_dialogue, action_items)

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download MoM PDF", f, file_name="meeting_mom.pdf", mime="application/pdf")
