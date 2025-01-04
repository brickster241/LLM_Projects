import streamlit as st
from llm_summarizer import LLM_Handler
from dotenv import load_dotenv
import os

# Set up the page configuration
st.set_page_config(
    page_title="LLM Summarizer",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Load environment variables (OPENAI API Key)
load_dotenv()

# Initialize session state for storing inputs
if "summary" not in st.session_state:
    st.session_state["summary"] = None

background_image_path =  os.path.join(os.getenv("ASSETS_PATH"), "background_streamlit.jpg")

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

# Title of the app
st.sidebar.title("🌟 LLM Summarizer 🌟")
st.sidebar.divider()

st.sidebar.write("""               
Welcome to the **LLM Summarizer**, an intuitive, AI-powered application designed to simplify and condense information from various media sources into concise summaries.  

🚀 **Features** :   
- 📄 **PDF Summarization**: Upload a custom PDF document, and the app extracts the text to provide a succinct summary.  
- 🎙️ **Audio Processing**: Upload `.wav` files to transcribe spoken content and generate a digestible summary.  
- 🎥 **YouTube Summarization**: Enter a YouTube video URL, and the tool fetches the transcript, condenses it, and provides key highlights.  
- ✂️ **Character Limit Control**: Summaries are tailored to fit your desired length, ensuring they remain concise and relevant.  
""")
st.sidebar.divider()

# Dropdown for input type selection
input_type = st.sidebar.selectbox(
    "Select Input Type :",
    ["📄 PDF", "🎙️ Audio", "🎥 YouTube"]
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

elif input_type == "🎥 YouTube":
    # Text input for YouTube URL
    youtube_url = st.sidebar.text_input("Enter the YouTube video URL :")
    if youtube_url:
        user_input = youtube_url
        st.sidebar.write(f"🎥 YouTube URL entered: {youtube_url}")

summary_size = st.sidebar.number_input("Maximum Words in Summary : ", min_value=75, max_value=500)

# Summarize button
if st.sidebar.button("Summarize"):
    # Create a new LLM Handler Object.
    llm_handler = LLM_Handler(summary_size)

    if user_input:

        # Audio
        if input_type == "🎙️ Audio":
            with st.spinner("Transcribing Audio File..."):
                try:
                    # Save the uploaded file temporarily
                    with open("temp_audio.wav", "wb") as f:
                        f.write(user_input.read())
                    
                    # Transcribe using Whisper
                    transcription = llm_handler.extract_text_from_audio(user_input.name)
                    
                    st.sidebar.success("Transcription complete !!")

                except Exception as e:
                    st.sidebar.error(f"Error during Audio transcription: {e}")
                finally:
                    # Clean up temporary file
                    if os.path.exists("temp_audio.wav"):
                        os.remove("temp_audio.wav")

        # PDF                
        elif input_type == "📄 PDF":
            with st.spinner("Parsing PDF File..."):
                try:
                    
                    # Parse using PyPDF2
                    pdf_text = llm_handler.extract_text_from_pdf(uploaded_pdf)
                    
                    st.sidebar.success("Parsing complete !!")
                    
                except Exception as e:
                    st.sidebar.error(f"Error during PDF Parsing: {e}")

        # Generate the summary (Placeholder logic)
        st.session_state["summary"] = f"Here is the summary for your {input_type} input!"
        st.title("📋 Summary")
        st.write(st.session_state.get("summary", "No summary available."))
    else:
        st.sidebar.warning("Please provide a valid input before summarizing.")

