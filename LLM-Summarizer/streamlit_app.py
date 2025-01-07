import streamlit as st
from llm_summarizer import LLM_Handler
from dotenv import load_dotenv
import os
import ollama

# Set up the page configuration
st.set_page_config(
    page_title="LLM Summarizer",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Load environment variables (OPENAI API Key)
load_dotenv()

# Add custom CSS for background customization
st.markdown(
    f"""
    <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {{
            background-image: linear-gradient(to bottom, #1c1c3a, #4b0082);
            color: white;
        }}

        /* Text color for better visibility */
        [data-testid="stSidebar"], .st-bq {{
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

llm_handler = LLM_Handler()

# Title of the app
st.sidebar.title("🌟 LLM Summarizer 🌟")
st.sidebar.divider()

st.sidebar.write("""
Welcome to the **LLM Summarizer**, an intuitive, AI-powered application designed to simplify and condense information from various media sources into concise summaries.  

🚀 **Features** :   
- 📄 **PDF Summarization**: Upload a custom PDF document, and the app extracts the text to provide a succinct summary.  
- 🎙️ **Audio Processing**: Upload `.wav` files to transcribe spoken content and generate a digestible summary.  
- 🎥 **Video Summarization**: Upload mp4 files, and the tool fetches the transcript, condenses it, and provides key highlights.  
- ✂️ **Character Limit Control**: Summaries are tailored to fit your desired length, ensuring they remain concise and relevant.  
- 🤖 **Multiple LLM Selection**: Choose from various AI models for summarization, including **GPT-4o-mini**, **qwen 2.5**, **llama3.2**, and **gemma**, to match your specific needs.  
""")

st.sidebar.divider()

# Dropdown for input type selection
input_type = st.sidebar.selectbox(
    "Select Input Type :",
    ["📄 PDF", "🎙️ Audio", "🎥 Video"]
)

# Options with icons
llm_options = {
    "gpt-4o-mini": "🤖 GPT-4o-Mini",
    "qwen2.5:latest": "🧠 Qwen 2.5",
    "llama3.2": "🦙 Llama 3.2",
    "gemma:7b": "✨ Gemma"
}

# Sidebar for LLM selection
selected_llm = st.sidebar.selectbox(
    "Select an LLM for summarization:",
    options=list(llm_options.values())
)

# Initialize input variable
user_input = None

# Conditional rendering based on input type
if input_type == "📄 PDF":
    # File uploader for PDF
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF File : ", type=["pdf"])
    if uploaded_pdf:
        user_input = uploaded_pdf
        st.sidebar.write("📄 PDF uploaded successfully!")

elif input_type == "🎙️ Audio":
    # File uploader for audio
    uploaded_audio = st.sidebar.file_uploader("Upload an .wav Audio File : ", type=["wav"])
    if uploaded_audio:
        user_input = uploaded_audio
        st.sidebar.write("🎵 Audio file uploaded successfully!")

elif input_type == "🎥 Video":
    # File Uploader for Video
    uploaded_video = st.sidebar.file_uploader("Upload a .mp4 Video File : ", type=["mp4"])
    if uploaded_video:
        user_input = uploaded_video
        st.sidebar.write(f"🎥 Video File uploaded successfully!")

summary_size = st.sidebar.number_input("Maximum Words in Summary : ", min_value=75, max_value=1000)

# Summarize button
if st.sidebar.button("Summarize"):
    # Create a new LLM Handler Object.
    transcription = None
    if user_input:

        # Audio
        if input_type == "🎙️ Audio":
            with st.spinner("Transcribing Audio File..."):

                # Transcribe using Whisper
                transcription = llm_handler.extract_text_from_audio(uploaded_audio)
                
                if transcription is not None:
                    st.toast("🎙️ Transcription complete !!")
                    st.sidebar.success("Audio Transcription complete !!")
                    st.session_state["summary"] = transcription
                else:
                    st.sidebar.error(f"Error during Audio transcription !!")
                    st.session_state["summary"] = None

        # PDF                
        elif input_type == "📄 PDF":
            with st.spinner("Parsing PDF File..."):
                # Parse using PyPDF2
                transcription = llm_handler.extract_text_from_pdf(uploaded_pdf)

                if transcription is not None:
                    st.toast("📄 Parsing complete !!")
                    st.sidebar.success("PDF Parsing complete !!")
                    st.session_state["summary"] = transcription
                else:
                    st.sidebar.error(f"Error during PDF Parsing !!")
                    st.session_state["summary"] = None
                           
        # Video               
        elif input_type == "🎥 Video":
            with st.spinner("Transcribing Video File..."):

                 # Transcribe using Whisper
                transcription = llm_handler.extract_text_from_audio(uploaded_video)
                
                if transcription is not None:
                    st.toast("🎙️ Transcription complete !!")
                    st.sidebar.success("Video Transcription complete !!")
                    st.session_state["summary"] = transcription
                else:
                    st.sidebar.error(f"Error during Video transcription !!")
                    st.session_state["summary"] = None

        # Generate the summary (Placeholder logic)
        try:
            with st.spinner("Summarizing Data..."):
                llm_model = [key for key, value in llm_options.items() if value == selected_llm][0]
                markdown_text = ""
                summary_st_placeholder = st.empty()
                for chunk in llm_handler.fetch_summary_from_transcription(transcription, summary_size, input_type, llm_model):
                    if chunk is not None:
                        markdown_text += chunk
                        summary_st_placeholder.markdown(body=markdown_text, unsafe_allow_html=True)
                st.success("Transcription Summary Complete !!")

        except Exception as e:
            st.toast(f"Error While Summarizing : {e}")
            print(f"Exception : {e}")
    else:
        st.sidebar.warning("Please provide a valid input before summarizing.")

