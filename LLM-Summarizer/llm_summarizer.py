# Import Libraries for Extracting Text from PDF, Youtube Video, Audio Files.

import os
import PyPDF2
import whisper

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

        # if not self.api_key:
        #     raise ValueError("OpenAI API Key not found in .env file.")

    ## Extracts text from a PDF file.
    def extract_text_from_pdf(self, pdf_file):
        
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    ## Extracts text from a Audio File.
    def extract_text_from_audio(self, audio_file_name):
        # Load the Whisper model
        model = whisper.load_model(self.WHISPER_MODEL)
        audio_file_path = os.path.join(self.ASSETS_PATH, audio_file_name)
        print(f"Audio File Path : {audio_file_path}")
        # Transcribe the .wav file
        result = model.transcribe(audio_file_path)

        return result["text"]
        
