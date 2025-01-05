# Import Libraries for Extracting Text from PDF, Youtube Video, Audio Files.

import os
import PyPDF2
import whisper
import uuid

"""
Python class implementation for LLM_Handler.
Includes methods for extracting text from PDFs, audio files, and YouTube videos.
""" 
class LLM_Handler:
    
    # Constructor , which stores the API Key.

    def __init__(self, summary_size):
        
        # Initialize OpenAI API key, and other variables
        
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.LLM_MODEL = "gpt-4o-mini"
        self.WHISPER_MODEL = os.getenv("WHISPER_MODEL")
        self.ASSETS_PATH = os.getenv("ASSETS_PATH")
        self.summary_size = summary_size

        if not self.API_KEY:
            raise ValueError("OpenAI API Key not found in .env file.")

    # Save the uploaded file to a temporary location and return the absolute path.
    def save_temp_file(self, uploaded_file):
        unique_filename = str(uuid.uuid4()) + os.path.splitext(uploaded_file.name)[1]
        with open(unique_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.abspath(unique_filename)

    ## Extracts text from a PDF file.
    def extract_text_from_pdf(self, pdf_file):
        
        try:

            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        except Exception as e:
            print(f"Exception during PDF Parsing : {e}")

    
    ## Extracts text from a .wav or .mp3 Audio File.
    def extract_text_from_audio(self, uploaded_audio_file):
        # Save the uploaded audio file to a temporary path
        temp_audio_path = self.save_temp_file(uploaded_audio_file)
            
        try:

            
            model = whisper.load_model(self.WHISPER_MODEL)
            
            # Perform transcription using Whisper
            result = model.transcribe(temp_audio_path)
            return result["text"]
        
        except Exception as e:
            print(f"Exception during Audio Transcription : {e}")
        
        finally: 
            # Clean up the temporary file
            os.remove(temp_audio_path)

    
    ## Extracts text from .mp4 Video File.
    def extract_text_from_video(self, uploaded_video_file):
        # Save the uploaded audio file to a temporary path
        temp_video_path = self.save_temp_file(uploaded_video_file)

        try:
            model = whisper.load_model(self.WHISPER_MODEL)
            
            # Perform transcription using Whisper
            result = model.transcribe(temp_video_path)
            return result["text"]  

        except Exception as e:
            print(f"Exception during Video Transcription : {e}")

        finally: 
            # Clean up the temporary file
            os.remove(temp_video_path)
        
        
