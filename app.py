# app.py
import os
import re
import json
import tempfile
from datetime import datetime
from collections import Counter, defaultdict

import streamlit as st
import whisper
import spacy
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm

# ---------------- CONFIG ----------------
MODEL_SIZE = "small"           # whisper model size (tiny/base/small/medium/large)
TIMELINE_SEGMENTS = 6         # how many timeline rows
SPEAKER_SUMMARY_MAX = 120     # tokens (approx) for speaker summarizer output length
# ----------------------------------------

st.set_page_config(page_title="MemoMaster Lite - MoM Generator", layout="wide")

st.title("🎙️ MemoMaster Lite — AI Minutes of Meeting (MoM)")

# ---------------- cached resources ----------------
@st.cache_resource(show_spinner=False)
def load_whisper(model_size=MODEL_SIZE):
    return whisper.load_model(model_size)

@st.cache_resource(show_spinner=False)
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner=False)
def load_summarizer():
    # smaller summarizer can be used to save memory; change model if needed
    return pipeline("summarization", model="facebook/bart-large-cnn")

whisper_model = load_whisper()
nlp = load_spacy()
summarizer = load_summarizer()

# ---------------- helpers ----------------
TASK_REGEX = r'(?:(?:To |Assign(?:ed to )?)?([A-Z][a-zA-Z]{1,40}))?.*?\b(submit|prepare|share|complete|send|deliver|provide|finalize|action|task)\b.*?(?:on|by)?\s*([0-9]{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]+)[-/ ]\d{2,4}|[0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})?'

def transcribe_audio_file(path):
    """
    Transcribe with whisper. If detected language not English, attempt translation.
    Returns: segments(list of dict {start,end,text}), detected_lang, full_text
    """
    result = whisper_model.transcribe(path, verbose=False)
    detected_lang = result.get("language", None)
    segments = result.get("segments", None) or []
    full_text = result.get("text", "").strip()

    # if detected non-English, try translate to English
    if detected_lang and detected_lang != "en":
        try:
            t = whisper_model.transcribe(path, task="translate", verbose=False)
            segments = t.get("segments", segments) or segments
            full_text = t.get("text", full_text).strip()
            detected_lang = "en"
        except Exception:
            # fallback to original
            pass

    # normalize segments to simple dicts
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

def build_timeline(segments, max_rows=TIMELINE_SEGMENTS):
    timeline = []
    for seg in segments[:max_rows]:
        start = seg.get("start", 0.0)
        end = seg.get("end", start)
        text = seg.get("text", "").replace("\n", " ").strip()
        topic = (text.split(".")[0])[:120]
        timeline.append({"start": start, "end": end, "topic": topic})
    return timeline

def fmt_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def extract_participants(full_text, segments, top_k=6):
    """
    Try to detect person names using spaCy across full_text and segment-level PERSON entities.
    Returns list of top detected person names.
    """
    names = []
    # full doc
    doc = nlp(full_text or " ")
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            names.append(ent.text.strip())
    # look in segments as well
    for seg in segments:
        doc = nlp(seg.get("text",""))
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                names.append(ent.text.strip())
    # frequency
    freq = [n for n, _ in Counter(names).most_common(top_k)]
    return freq

def group_segments_by_speaker(segments, participants):
    """
    Heuristic speaker assignment:
    - If a PERSON entity appears in a segment, assign to that person.
    - Else if a participant name substring appears, assign.
    - Else assign to 'Unknown'
    Returns dict: speaker -> list of texts
    """
    speaker_texts = defaultdict(list)
    lower_participants = [p.lower() for p in participants]
    for seg in segments:
        text = seg.get("text","").strip()
        assigned = None
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                assigned = ent.text.strip()
                break
        if not assigned:
            for p, lp in zip(participants, lower_participants):
                if lp and lp in text.lower():
                    assigned = p
                    break
        if not assigned:
            assigned = "Unknown"
        speaker_texts[assigned].append(text)
    return speaker_texts

def summarize_text(text, max_length=130, min_length=20):
    """
    Wraps summarizer; splits into smaller chunks if too long.
    """
    if not text.strip():
        return ""
    # huggingface summarizer usually expects <1024 tokens; chunk if needed by sentences
    if len(text.split()) < 250:
        try:
            out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return out[0]["summary_text"]
        except Exception:
            return text
    # chunk
    sentences = text.split(". ")
    chunks = []
    current = ""
    for s in sentences:
        if len((current + s).split()) > 220:
            chunks.append(current.strip())
            current = s + ". "
        else:
            current += s + ". "
    if current.strip():
        chunks.append(current.strip())
    summaries = []
    for c in chunks:
        try:
            o = summarizer(c, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(o[0]["summary_text"])
        except Exception:
            summaries.append(c)
    return " ".join(summaries)

def extract_action_items(segments):
    """
    Extract action items and deadlines using regex on each segment sentence.
    Returns list of tuples: (Assignee, TaskText, Deadline)
    """
    action_items = []
    for seg in segments:
        text = seg.get("text","").strip()
        if not text:
            continue
        # split into sentences
        sentences = re.split(r'(?<=[\.\?\!])\s+', text)
        for s in sentences:
            m = re.search(TASK_REGEX, s, re.I)
            if m:
                assignee = (m.group(1) or "Unassigned").strip() if m.lastindex and m.lastindex>=1 else "Unassigned"
                # deadline may be in group 3
                deadline = m.group(3) if m.lastindex and m.lastindex>=3 else ""
                action_items.append((assignee, s.strip(), deadline or ""))
    return action_items

def generate_meeting_id():
    now = datetime.now()
    return f"MOM-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"

def write_mom_pdf(path, meeting_date, meeting_time, location, participants, meeting_id, timeline, meeting_summary, speaker_summaries, action_items):
    doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=10*mm, leftMargin=10*mm, topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodySmall', parent=styles['Normal'], fontSize=9, leading=11, alignment=0))
    styles.add(ParagraphStyle(name='HeadingCenter', parent=styles['Heading2'], fontSize=12, leading=14, alignment=1))

    story = []
    story.append(Paragraph("<b>Minutes of Meeting (MoM)</b>", styles['HeadingCenter']))
    story.append(Spacer(1,6))

    meta = f"<b>Meeting Date:</b> {meeting_date} &nbsp;&nbsp; <b>Time:</b> {meeting_time} &nbsp;&nbsp; <b>Location:</b> {location}"
    story.append(Paragraph(meta, styles['BodySmall']))
    story.append(Paragraph(f"<b>Participants:</b> {', '.join(participants) if participants else 'N/A'}", styles['BodySmall']))
    story.append(Paragraph(f"<b>Meeting ID:</b> {meeting_id}", styles['BodySmall']))
    story.append(Spacer(1,8))

    # Timeline
    story.append(Paragraph("<b>Meeting Timeline</b>", styles['BodySmall']))
    if timeline:
        table_data = [["Start","End","Topic"]]
        for row in timeline:
            table_data.append([fmt_time(row["start"]), fmt_time(row["end"]), row["topic"]])
        table = Table(table_data, colWidths=[50,50,420])
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#4F81BD")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("ALIGN",(0,0),(-1,-1),"LEFT"),
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("FONTSIZE",(0,0),(-1,-1),9)
        ]))
        story.append(table)
    story.append(Spacer(1,8))

    # Meeting Summary
    story.append(Paragraph("<b>Meeting Summary</b>", styles['BodySmall']))
    story.append(Paragraph(meeting_summary or "N/A", styles['BodySmall']))
    story.append(Spacer(1,8))

    # Speaker-wise Summarization
    story.append(Paragraph("<b>Speaker-wise Summarization</b>", styles['BodySmall']))
    for spk, summ in speaker_summaries.items():
        story.append(Paragraph(f"<b>{spk}:</b> {summ}", styles['BodySmall']))
    story.append(Spacer(1,8))

    # Action Items
    if action_items:
        story.append(Paragraph("<b>Action Items</b>", styles['BodySmall']))
        table_data = [["Assigned To","Task","Deadline"]]
        for a in action_items:
            table_data.append([a[0], a[1], a[2]])
        table = Table(table_data, colWidths=[120,380,80])
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#4F81BD")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.whitesmoke),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("ALIGN",(0,0),(-1,-1),"LEFT"),
            ("GRID",(0,0),(-1,-1),0.25,colors.grey),
            ("FONTSIZE",(0,0),(-1,-1),9)
        ]))
        story.append(table)

    doc.build(story)

# ---------------- Streamlit flow ----------------
st.markdown("Upload meeting audio (mp3 / wav / m4a). The app will transcribe, summarize, extract participants and action items, and generate a formatted MoM PDF.")

uploaded = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])

if uploaded:
    # Save uploaded to a tmp file (whisper accepts path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tf:
        tf.write(uploaded.read())
        temp_audio_path = tf.name

    st.info("Transcribing audio. This can take a while on first run (models download).")
    with st.spinner("Transcribing..."):
        segments, lang, full_text = transcribe_audio_file(temp_audio_path)

    st.subheader("📝 Transcript (first 1500 chars)")
    st.write(full_text[:1500] + ("..." if len(full_text) > 1500 else ""))

    # timeline
    timeline = build_timeline(segments, max_rows=TIMELINE_SEGMENTS)

    # participants
    participants = extract_participants(full_text, segments, top_k=6)

    # group segments by speaker
    speaker_texts = group_segments_by_speaker(segments, participants)

    # speaker summarization
    speaker_summaries = {}
    for spk, texts in speaker_texts.items():
        joined = " ".join(texts)
        speaker_summaries[spk] = summarize_text(joined, max_length=60, min_length=10) or (joined[:200] + ("..." if len(joined) > 200 else ""))

    # meeting summary (summarize full_text)
    with st.spinner("Creating meeting summary..."):
        meeting_summary = summarize_text(full_text, max_length=200, min_length=40)

    # action items
    action_items = extract_action_items(segments)

    # heuristics for date/time/location
    date_match = re.search(r'(\b\d{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]+)[-/ ]\d{2,4}\b)', full_text, re.I)
    meeting_date = date_match.group(1) if date_match else datetime.now().strftime("%d-%b-%Y")
    time_match = re.search(r'(\b\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)?)', full_text)
    meeting_time = time_match.group(1) if time_match else "Unknown"
    location = "Virtual (Zoom)" if re.search(r'\bzoom|teams|meet|virtual\b', full_text, re.I) else "Virtual"

    meeting_id = generate_meeting_id()

    st.subheader("📌 Extracted Summary")
    st.write(meeting_summary)

    st.subheader("👥 Participants (detected)")
    st.write(", ".join(participants) if participants else "None detected")

    st.subheader("🗂 Action Items (detected)")
    if action_items:
        for a in action_items:
            st.write(f"• **{a[0]}** — {a[1]} — _{a[2]}_")
    else:
        st.write("No action items detected.")

    # Generate PDF
    pdf_filename = f"{os.path.splitext(uploaded.name)[0]}_MoM.pdf"
    pdf_path = os.path.join(tempfile.gettempdir(), pdf_filename)
    write_mom_pdf(pdf_path, meeting_date, meeting_time, location, participants, meeting_id, timeline, meeting_summary, speaker_summaries, action_items)

    st.success("MoM PDF ready ✅")
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download MoM PDF", f, file_name=pdf_filename, mime="application/pdf")

    # cleanup
    try:
        os.remove(temp_audio_path)
    except Exception:
        pass
