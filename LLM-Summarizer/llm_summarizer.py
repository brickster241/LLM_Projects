# Import Libraries for Extracting Text from PDF, Youtube Video, Audio Files.

import os
import PyPDF2
import whisper
import uuid
from openai import OpenAI

"""
Python class implementation for LLM_Handler.
Includes methods for extracting text from PDFs, audio files, and YouTube videos.
""" 
class LLM_Handler:
    
    # Constructor , which stores the API Key.

    def __init__(self):
        
        # Initialize OpenAI API key, and other variables
        
        self.API_KEY = os.getenv("OPENAI_API_KEY")
        self.LLM_MODEL = "gpt-4o-mini"
        self.WHISPER_MODEL = "base"

        if not self.API_KEY:
            raise ValueError("OpenAI API Key not found in .env file.")
        else:
            self.openai = OpenAI()

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

    ## Connect to OpenAI's Model and fetch summary in Markdown Format.
    def fetch_summary_from_transcription(self, transcription_text, summary_size, input_type):
        if transcription_text is None:
            return ""
        
        # system prompt -> that tells them what task they are performing and what tone they should use.
        # user prompt -> the conversation starter that they should reply to.

        system_prompt = f"""You are a professional summarizer. Summarize the provided transcription into a structured
          format. Use headings and bullet points to organize the summary, capturing the main ideas
          , critical points, and key takeaways for easy readability. Respond the formatted summary in Markdown."""

        user_prompt = f"""Please summarize the {input_type} transcription text below into a Markdown-formatted structured summary
          of no more than {summary_size} words. Highlight the main topics, critical points, and key takeaways using headings and bullet points.
          Transcription : {transcription_text}"""

        response = self.openai.chat.completions.create(model = "gpt-4o-mini",
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ])
        return response.choices[0].message.content
        
