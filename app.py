import streamlit as st
import whisper
import tempfile
import os
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Title
st.title("🎙️ MemoMaster Lite - AI Minutes of Meeting Generator")

# Whisper model (base for speed, can switch to "small", "medium", "large")
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

# Hugging Face summarizer
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")
    
summarizer = load_summarizer()

# Upload audio file
uploaded_file = st.file_uploader("Upload a meeting audio file (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    st.info("⏳ Transcribing audio... please wait.")

    # Transcribe with Whisper
    result = whisper_model.transcribe(temp_path)
    transcript = result["text"]

    st.subheader("📝 Full Transcript")
    st.write(transcript)

    # Summarize transcript
    st.info("⏳ Generating meeting summary...")
    summary_chunks = summarizer(transcript, max_length=130, min_length=30, do_sample=False)
    meeting_summary = summary_chunks[0]['summary_text']

    st.subheader("📌 Meeting Summary")
    st.write(meeting_summary)

    # Generate PDF
    pdf_path = "meeting_summary.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Minutes of Meeting")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, "Transcript:")
    text_obj = c.beginText(50, height - 120)
    text_obj.setFont("Helvetica", 10)
    for line in transcript.split(". "):
        text_obj.textLine(line.strip())
    c.drawText(text_obj)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 400, "Summary:")
    text_obj = c.beginText(50, height - 420)
    text_obj.setFont("Helvetica", 10)
    for line in meeting_summary.split(". "):
        text_obj.textLine(line.strip())
    c.drawText(text_obj)

    c.save()

    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="⬇️ Download Meeting Summary (PDF)",
            data=pdf_file,
            file_name="meeting_summary.pdf",
            mime="application/pdf"
        )

    # Cleanup
    os.remove(temp_path)
