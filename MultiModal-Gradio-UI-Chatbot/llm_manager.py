from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import os
from dotenv import load_dotenv

class LLM_Manager:
    """
    A manager class to interact with various Large Language Model (LLM) providers: 
    OpenAI, Anthropic (Claude), and Google (Gemini).
    This class handles the initialization of API clients and provides methods to query each LLM.
    """
    
    def __init__(self):
        """
        Constructor to initialize API clients for OpenAI, Anthropic, and Google Generative AI.
        API keys are loaded from a `.env` file.
        """
        # Load API keys from the .env file
        load_dotenv()
        
        self._openai_api_key = os.getenv('OPENAI_API_KEY')
        self._anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self._google_api_key = os.getenv('GOOGLE_API_KEY')

        self.openai_client = OpenAI()
        self.claude_client = Anthropic()
        self.gemini_client = None
        genai.configure()

    def query_openai(self, system_prompt, msg_prompts, max_tokens=200, temperature=0.7, is_stream=False):
        """
        Queries OpenAI's GPT model with the given prompts and configuration.

        Args:
            system_prompt (str): Initial system-level instruction for the model.
            msg_prompts (list): List of messages or prompts to send to the model.
            max_tokens (int): Maximum number of tokens to generate in the response.
            temperature (float): Sampling temperature to control response randomness.
            is_stream (bool): If True, enables streaming of responses.

        Returns:
            str or generator: Generated response content, or a stream of content if is_stream is True.
        """
        response = self.openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=msg_prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=is_stream
        )
        if not is_stream:
            return response.choices[0].message.content
        else:
            for chunk in response:
                yield chunk.choices[0].delta.content

    def query_claude(self, system_prompt, msg_prompts, max_tokens=200, temperature=0.7, is_stream=False):
        """
        Queries Anthropic's Claude model with the given prompts and configuration.

        Args:
            system_prompt (str): Initial system-level instruction for the model.
            msg_prompts (list): List of messages or prompts to send to the model.
            max_tokens (int): Maximum number of tokens to generate in the response.
            temperature (float): Sampling temperature to control response randomness.
            is_stream (bool): If True, enables streaming of responses.

        Returns:
            str or generator: Generated response content, or a stream of content if is_stream is True.
        """
        # API needs system message provided separately from user prompt
        response = None
        if not is_stream:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=msg_prompts
            )
            return response.content[0].text
        else:
            response = self.claude_client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=msg_prompts
            )
            with response as stream:
                for text in stream.text_stream:
                    yield text

    def query_gemini(self, system_prompt, msg_prompts, is_stream=False):
        """
        Queries Google's Gemini model with the given prompts and configuration.

        Args:
            system_prompt (str): Initial system-level instruction for the model.
            msg_prompts (list): List of messages or prompts to send to the model.
            is_stream (bool): If True, enables streaming of responses.

        Returns:
            str or generator: Generated response content, or a stream of content if is_stream is True.
        """
        self.gemini_client = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=system_prompt
        )
        response = self.gemini_client.generate_content(
            stream=is_stream,
            contents=msg_prompts
        )
        if is_stream:
            for chunk in response:
                yield chunk.text
        else:
            return response.text
