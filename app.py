import os, re, json, tempfile
from datetime import datetime
from collections import defaultdict

import streamlit as st
import whisper
import spacy
import en_core_web_sm
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
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
    return en_core_web_sm.load()

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

whisper_model = load_whisper_model()
nlp = load_spacy_model()
summarizer = load_summarizer()

# ---------------- HELPERS ----------------
def transcribe_and_translate_if_needed(audio_path):
    result = whisper_model.transcribe(audio_path, verbose=False)
    detected_lang = result.get("language", None)
    segments = result.get("segments", None) or []
    full_text = result.get("text", "").strip()

    if not segments and full_text:
        segments = [{"start": 0.0, "end": 0.0, "text": full_text}]

    if detected_lang and detected_lang != "en":
        try:
            translated = whisper_model.transcribe(audio_path, task="translate", verbose=False)
            t_segments = translated.get("segments", None)
            t_text = translated.get("text", "").strip()
            if t_segments:
                conv = []
                for s in t_segments:
                    conv.append({"start": s.get("start", 0.0), "end": s.get("end", 0.0), "text": s.get("text", "").strip()})
                segments = conv
                full_text = t_text
            else:
                full_text = t_text or full_text
                if full_text and not segments:
                    segments = [{"start": 0.0, "end": 0.0, "text": full_text}]
            detected_lang = "en"
        except Exception as e:
            st.warning(f"[!] Translation failed: {e}")

    return segments, detected_lang, full_text

TASK_REGEX = r'(?:(?:To |Assign(ed to )?)?([A-Z][a-zA-Z]{1,30}))?.*?\b(submit|prepare|share|complete|send|deliver|provide|finalize)\b.*?\b(on|by)\b\s+([0-9]{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]+)[-/ ]\d{2,4}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})'

def extract_key_points_with_deadlines(segments):
    action_items = []
    for seg in segments:
        text = seg.get("text","").strip()
        if not text:
            continue
        doc = nlp(text)
        for sent in doc.sents:
            s = sent.text.strip()
            m = re.search(TASK_REGEX, s, re.I)
            if m:
                assignee = (m.group(2) or "Unassigned") if m.groups() else "Unassigned"
                deadline = m.group(5) if m.lastindex and m.lastindex>=5 else ""
                action_items.append((assignee, s, deadline))
    return action_items

def build_timeline(segments, max_rows=6):
    timeline = []
    for seg in segments[:max_rows]:
        start = seg.get("start",0.0); end = seg.get("end", start)
        text = seg.get("text","").replace("\n"," ").strip()
        topic = (text.split(".")[0])[:80] if text else ""
        timeline.append({"start":start, "end":end, "topic": topic})
    return timeline

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
    if speaker_dialogue:
        story.append(Paragraph("<b>Speaker-wise Dialogue</b>", styles['BodySmall']))
        for speaker, lines in speaker_dialogue.items():
            story.append(Paragraph(f"<b>{speaker}</b>", styles['BodySmall']))
            for l in lines:
                story.append(Paragraph(f"• {l}", styles['BodySmall']))
        story.append(Spacer(1,6))

    # Action items
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

def generate_summary(full_text):
    try:
        out = summarizer(full_text, max_length=200, min_length=60, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return full_text

def generate_mom_from_audio_file(audio_file):
    base = os.path.splitext(os.path.basename(audio_file))[0]
    with tempfile.TemporaryDirectory() as td:
        segments, detected_lang, full_text = transcribe_and_translate_if_needed(audio_file)
        timeline = build_timeline(segments, max_rows=TIMELINE_SEGMENTS)
        action_items = extract_key_points_with_deadlines(segments)

        speaker_dialogue = defaultdict(list)
        participants = []
        for seg in segments:
            text = seg.get("text","").strip()
            assigned = None
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    assigned = ent.text
                    break
            assigned = assigned or "Unknown"
            speaker_dialogue[assigned].append(text)
            if assigned not in participants:
                participants.append(assigned)

        summary_text = generate_summary(full_text)
        meeting_date = datetime.now().strftime("%d-%b-%Y")
        meeting_time = "10:00 AM - 11:00 AM"
        location = "Virtual"

        pdf_path = base + "_MoM.pdf"
        write_mom_pdf(pdf_path, meeting_date, meeting_time, location, participants, timeline, summary_text, speaker_dialogue, action_items)
        return pdf_path

# ---------------- STREAMLIT UI ----------------
st.title("📑 AI Minutes of Meeting (MoM) Generator")
uploaded_file = st.file_uploader("Upload a meeting audio file", type=["mp3","wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    with st.spinner("Processing audio..."):
        pdf_path = generate_mom_from_audio_file(audio_path)

    st.success("✅ MoM generated successfully!")
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download MoM PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
