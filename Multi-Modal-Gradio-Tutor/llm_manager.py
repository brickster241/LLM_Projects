from openai import OpenAI
import ollama
import google.generativeai as genai
import os
from dotenv import load_dotenv

class LLM_Manager:
    """
    A manager class to interact with various Large Language Model (LLM) providers: 
    OpenAI, Ollama, and Google (Gemini).
    This class handles the initialization of API clients and provides methods to query each LLM.
    """
    
    def __init__(self):
        """
        Constructor to initialize API clients for OpenAI, Google Generative AI.
        API keys are loaded from a `.env` file.
        """
        # Load API keys from the .env file
        load_dotenv()
        
        self._openai_api_key = os.getenv('OPENAI_API_KEY')
        self._google_api_key = os.getenv('GOOGLE_API_KEY')

        self.openai_client = OpenAI()
        self.gemini_client = None
        genai.configure()

    def query_llama(self, system_prompt, msg_prompts):
        """
        Queries Ollama's Llama3.2 model with the given prompts and configuration.

        Args:
            system_prompt (str): Initial system-level instruction for the model.
            msg_prompts (list): List of messages or prompts to send to the model, except the system prompt.
        Returns:
            string: Generated content.
        """
        response = ollama.chat(model="llama3.2:3b", 
            messages=msg_prompts,
            stream=False)
        return response.message.content


    def query_gemini(self, system_prompt, msg_prompts):
        """
        Queries Google's Gemini-1.5-Flash model with the given prompts and configuration.

        Args:
            system_prompt (str): Initial system-level instruction for the model.
            msg_prompts (list): List of messages or prompts to send to the model.

        Returns:
            string: Generated content.
        """
        self.gemini_client = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=system_prompt
        )
        response = self.gemini_client.generate_content(
            stream=False,
            contents=msg_prompts
        )
        return response.text