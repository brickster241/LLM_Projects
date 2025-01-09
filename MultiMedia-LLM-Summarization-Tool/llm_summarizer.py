# Import Libraries for Extracting Text from PDF, Youtube Video, Audio Files.

import os
import PyPDF2
import whisper
import uuid
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import ollama

"""
Python class implementation for LLM_Handler.
Includes methods for extracting text from PDFs, audio files, and YouTube videos.
""" 
class LLM_Handler:
    
    # Constructor , which stores the API Key.

    def __init__(self):
        
        # Initialize OpenAI API key, and other variables
        
        self._openai_api_key = os.getenv('OPENAI_API_KEY')
        self._anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self._google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openai_client = None
        self.claude_client = None
        self.gemini_client = None

        if not self._openai_api_key:
            raise ValueError("OpenAI API Key not found in .env file.")
        else:
            self.openai_client = OpenAI()
        if not self._anthropic_api_key:
            raise ValueError("Anthropic API Key not found in .env file.")
        else:
            self.claude_client = Anthropic()
        if not self._google_api_key:
            raise ValueError("Google API Key not found in .env file.")
        else:
            genai.configure()
        
        self.WHISPER_MODEL = "base"

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
    def fetch_summary_from_transcription(self, transcription_text, summary_size, input_type, llm_model):
        if transcription_text is None:
            return ""
        
        # system prompt -> that tells them what task they are performing and what tone they should use.
        # user prompt -> the conversation starter that they should reply to.

        system_prompt = f"""You are a professional summarizer. Summarize the provided transcription into a structured format
          using natural language. Organize the summary under headings and subheadings with flowing, coherent sentences. 
          Use bullet points only where necessary, ensuring the text is readable and well-connected. 
          Prioritize clarity and a logical flow of ideas, capturing the main points, critical details, and actionable takeaways.
         Deliver the summary in Markdown format.
"""

        user_prompt = f"""Please summarize the {input_type} transcription text below into a structured summary
          of no more than {summary_size} words. Use bullet points only where necessary, ensuring the text is readable and well-connected. 
          Prioritize clarity and a logical flow of ideas, capturing the main points, critical details, and actionable takeaways. Respond in markdown format. Transcription : {transcription_text}"""
        
        response = None
        # Check Model Type
        if llm_model == "gpt-4o-mini":
            response = self.openai_client.chat.completions.create(model = "gpt-4o-mini",
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ], stream=True)
            for chunk in response:
                yield chunk.choices[0].delta.content

        elif llm_model == "claude-3-haiku-20240307":
            response = self.claude_client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=summary_size,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            with response as stream:
                for text in stream.text_stream:
                    yield text

        elif llm_model == "gemini-1.5-flash":
            self.gemini_client = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                system_instruction=system_prompt
            )
            response = self.gemini_client.generate_content(
                stream=True,
                contents=[
                    {"role": "user", "parts": user_prompt}
                ]
            )
            for chunk in response:
                yield chunk.text

        else:
            response = ollama.chat(model=llm_model, messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ], stream=True)
            for chunk in response:
                yield chunk.message.content
        
