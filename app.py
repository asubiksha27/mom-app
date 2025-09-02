import os, re, json, tempfile
from datetime import datetime
from collections import defaultdict, Counter

import streamlit as st
import cv2, pytesseract, whisper, spacy
from moviepy.editor import VideoFileClip
from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# ---------------- CONFIG ----------------
MODEL_SIZE = "small"
OCR_INTERVAL_SEC = 4
TIMELINE_SEGMENTS = 6

# ---------- your original helpers ----------
def safe_extract_audio(video_path, audio_path):
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        clip.close()
        raise RuntimeError("Video has no audio track.")
    clip.audio.write_audiofile(audio_path, fps=16000, codec='pcm_s16le', verbose=False, logger=None)
    clip.close()
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) < 4000:
        raise RuntimeError("Audio extraction failed (file too small).")

def extract_visual_names_from_video(video_path, interval_sec=4, max_names=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(fps * interval_sec))
    names, cnt = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cnt % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            crop = gray[int(h*0.75):h, 0:w]
            text = pytesseract.image_to_string(crop)
            for line in text.splitlines():
                s = line.strip()
                if 1 < len(s) <= 40 and re.search(r'[A-Za-z]', s):
                    names.append(s)
        cnt += 1
    cap.release()
    from collections import Counter
    return [n for n, _ in Counter(names).most_common(max_names)]

def transcribe_and_translate_if_needed(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, verbose=False)
    detected_lang = result.get("language", None)
    segments = result.get("segments", None) or []
    full_text = result.get("text", "").strip()
    if not segments and full_text:
        segments = [{"start": 0.0, "end": 0.0, "text": full_text}]
    if detected_lang and detected_lang != "en":
        try:
            translated = model.transcribe(audio_path, task="translate", verbose=False)
            t_segments = translated.get("segments", None)
            t_text = translated.get("text", "").strip()
            if t_segments:
                conv = []
                for s in t_segments:
                    conv.append({"start": s.get("start",0.0), "end": s.get("end",0.0), "text": s.get("text","").strip()})
                segments = conv
                full_text = t_text
            else:
                full_text = t_text or full_text
                if full_text and not segments:
                    segments = [{"start":0.0,"end":0.0,"text":full_text}]
            detected_lang = "en"
        except Exception as e:
            st.warning(f"[!] Translation failed: {e}")
    uniform = []
    for s in segments:
        uniform.append({"start": s.get("start",0.0), "end": s.get("end",0.0), "text": s.get("text","")})
    return uniform, detected_lang, full_text

nlp = spacy.load("en_core_web_sm")
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
                deadline = m.group(5) if m.lastindex and m.lastindex>=5 else (m.group(4) if m.lastindex and m.lastindex>=4 else "")
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
    story.append(Paragraph("<b>Meeting Timeline</b>", styles['BodySmall']))
    if timeline:
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
    story.append(Paragraph("<b>Meeting Summary</b>", styles['BodySmall']))
    story.append(Paragraph(summary_text, styles['BodySmall']))
    story.append(Spacer(1,6))
    story.append(Paragraph("<b>Speaker-wise Dialogue</b>", styles['BodySmall']))
    for speaker, lines in speaker_dialogue.items():
        story.append(Paragraph(f"<b>{speaker}</b>", styles['BodySmall']))
        for l in lines:
            story.append(Paragraph(f"• {l}", styles['BodySmall']))
    story.append(Spacer(1,6))
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

def generate_full_summary(full_text):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        out = summarizer(full_text, max_length=800, min_length=150, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return full_text

def generate_mom_from_video_file(video_file, model_size=MODEL_SIZE, ocr_interval_sec=OCR_INTERVAL_SEC):
    base = os.path.splitext(os.path.basename(video_file))[0]
    with tempfile.TemporaryDirectory() as td:
        audio_path = os.path.join(td, "audio.wav")
        safe_extract_audio(video_file, audio_path)
        visual_names = extract_visual_names_from_video(video_file, interval_sec=ocr_interval_sec)
        segments, detected_lang, full_text = transcribe_and_translate_if_needed(audio_path, model_size=model_size)
        timeline = build_timeline(segments, max_rows=TIMELINE_SEGMENTS)
        action_items = extract_key_points_with_deadlines(segments)
        speaker_dialogue = defaultdict(list)
        participants = list(visual_names)
        for seg in segments:
            text = seg.get("text","").strip()
            assigned = None
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    assigned = ent.text
                    break
            if not assigned:
                for vn in visual_names:
                    if vn.lower() in text.lower():
                        assigned = vn
                        break
            speaker_dialogue[assigned or "Unknown"].append(text)
            if (assigned or "Unknown") not in participants:
                participants.append(assigned or "Unknown")
        summary_text = generate_full_summary(full_text)
        date_match = re.search(r'(\b\d{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|[A-Za-z]+)[-/ ]\d{2,4}\b)', full_text, re.I)
        meeting_date = date_match.group(1) if date_match else datetime.now().strftime("%d-%b-%Y")
        time_match = re.search(r'(\b\d{1,2}[:.]\d{2}\s*(?:AM|PM|am|pm)?)', full_text)
        meeting_time = time_match.group(1) + " - " + (time_match.group(1) if time_match else "Unknown") if time_match else "10:00 AM - 11:00 AM"
        location = "Virtual (Zoom)" if re.search(r'\bzoom|teams|meet|virtual\b', full_text, re.I) else "Virtual"
        pdf_path = base + "_MoM.pdf"
        write_mom_pdf(pdf_path, meeting_date, meeting_time, location, participants, timeline, summary_text, speaker_dialogue, action_items)
        txt_path = base + "_MoM_debug.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Detected visual names:\n" + json.dumps(visual_names, indent=2, ensure_ascii=False) + "\n\n")
            f.write("Action items:\n" + json.dumps(action_items, indent=2, ensure_ascii=False) + "\n\n")
            f.write("Speaker dialogue:\n" + json.dumps({k:v[:5] for k,v in speaker_dialogue.items()}, indent=2, ensure_ascii=False) + "\n\n")
        return {"pdf": pdf_path, "txt": txt_path}

# ---------------- Streamlit UI ----------------
st.title("📑 AI Minutes of Meeting (MoM) Generator")
uploaded_file = st.file_uploader("Upload a meeting video", type=["mp4","mkv","mov"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    with st.spinner("Processing video..."):
        out = generate_mom_from_video_file(video_path)
    st.success("Done!")
    with open(out["pdf"], "rb") as f:
        st.download_button("📥 Download MoM PDF", f, file_name=os.path.basename(out["pdf"]), mime="application/pdf")
    with open(out["txt"], "rb") as f:
        st.download_button("📥 Download Debug TXT", f, file_name=os.path.basename(out["txt"]), mime="text/plain")
