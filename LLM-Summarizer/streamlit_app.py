import streamlit as st
from llm_summarizer import LLM_Handler
from dotenv import load_dotenv
import os

# Set up the page configuration
st.set_page_config(
    page_title="LLM Summarizer",
    page_icon="🌌",
    layout="centered"
)

# Load environment variables (OPENAI API Key)
load_dotenv()

# Initialize session state for storing inputs
if "summary" not in st.session_state:
    st.session_state["summary"] = None

# Title of the app
st.title("🌌 LLM Summarizer")

# Dropdown for input type selection
input_type = st.selectbox(
    "Select the type of input you want to summarize:",
    ["PDF", "Audio", "Website", "YouTube"]
)

# Initialize input variable
user_input = None

# Conditional rendering based on input type
if input_type == "PDF":
    # File uploader for PDF
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf:
        user_input = uploaded_pdf
        st.write("📄 PDF uploaded successfully!")

elif input_type == "Audio":
    # File uploader for audio
    uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])
    if uploaded_audio:
        user_input = uploaded_audio
        st.write("🎵 Audio file uploaded successfully!")

elif input_type == "YouTube":
    # Text input for YouTube URL
    youtube_url = st.text_input("Enter the YouTube video URL:")
    if youtube_url:
        user_input = youtube_url
        st.write(f"🎥 YouTube URL entered: {youtube_url}")

# Summarize button
if st.button("Summarize"):
    # Create a new LLM Handler Object.
    llm_handler = LLM_Handler()

    if user_input:

        # Audio
        if input_type == "Audio":
            with st.spinner("Transcribing audio..."):
                try:
                    # Save the uploaded file temporarily
                    with open("temp_audio.wav", "wb") as f:
                        f.write(user_input.read())
                    
                    # Transcribe using Whisper
                    transcription = llm_handler.extract_text_from_audio(user_input.name)
                    
                    st.success("Transcription complete!")
                    st.text_area("Transcription Result", transcription, height=200)
                except Exception as e:
                    st.error(f"Error during transcription: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists("temp_audio.wav"):
                        os.remove("temp_audio.wav")

        # Generate the summary (Placeholder logic)
        st.session_state["summary"] = f"Here is the summary for your {input_type} input!"
        st.title("📋 Summary Results")
        st.write(st.session_state.get("summary", "No summary available."))
    else:
        st.warning("Please provide a valid input before summarizing.")

