from llm_manager import LLM_Manager
import openai
import gradio as gr
import os
import subprocess
from io import BytesIO
from pydub import AudioSegment
import json

class MultiModal_Gradio_Assistant:
    """
    An AI assistant which leverages expertise from other sources for you.

    Features:
    - Multimodal: Can process and generate responses in multiple formats (text, voice, etc.).
    - Uses tools: Utilizes external tools to enhance its capabilities.
    - Streams responses: Provides responses incrementally for a better user experience.
    - Reads out the responses after streaming: Converts text responses to speech.
    - Converts voice to text during input: Transcribes voice input to text for processing.
    """

    def __init__(self):
        """
        Initializes the MultiModal_Gradio_Assistant instance and sets up necessary configurations.
        """
        # Directory for temporary audio files
        self.temp_dir = os.path.join(os.path.dirname(__file__), "TempAudio")

        # Whisper model for audio transcription
        self.whisper_model = "base"

        # LLM manager instance for handling LLM-related tasks
        self.llm_manager = LLM_Manager()

        # Ensure the temporary directory exists
        os.makedirs(self.temp_dir, exist_ok=True)

        # Define tools and variables
        self.define_tool_functions()
        self.define_assistant_variables()

    def define_assistant_variables(self):
        """
        Defines system prompts and tool mappings for the assistant.
        """
        # General behavior prompt for all models
        general_prompt = (
            "Please be as technical as possible with your answers. "
            "Only answer questions about topics you have expertise in. If you do not know something say so."
        )

        # Additional behavior-specific prompts for GPT
        additional_prompt_gpt = (
            "Analyze the user query and determine if the content is primarily related to coding, software engineering, "
            "data science and LLMs. If so please answer it yourself else if it is primarily related to Sports, Movies "
            "or Games, get answers from tool ask_gemini or if it belongs to none of them, get answers from tool ask_ollama. "
        )

        # Define system prompts for different tools
        self.sys_prompt_gpt = (
            "You are a helpful technical tutor who is an expert in coding, software engineering, data science and LLMs."
            + additional_prompt_gpt + general_prompt
        )
        self.sys_prompt_gemini = (
            "You are a helpful AI Assistant coach who is an expert in sports, movies and games."
            + general_prompt
        )
        self.sys_prompt_ollama = (
            "You are a helpful AI Assistant who is an expert in general knowledge, space and spoken languages."
            + general_prompt
        )

        # Define tools for assistant functionality
        self.assistant_tools = [
            {"type": "function", "function": self.ask_ollama_function},
            {"type": "function", "function": self.ask_gemini_function},
        ]

        # Mapping for tools and corresponding functions
        self.tools_functions_map = {
            "ask_ollama": self.ask_ollama,
            "ask_gemini": self.ask_gemini,
        }

    def define_tool_functions(self):
        """
        Defines metadata for tools available to the assistant.
        """
        self.ask_ollama_function = {
            "name": "ask_ollama",
            "description": (
                "Get the answer to the question related to a topic this agent is familiar with. Call this whenever "
                "you need to answer something related to general knowledge, space or languages. For example 'Teach me "
                "basics of Spanish?' or 'How big is our universe?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question_for_topic": {
                        "type": "string",
                        "description": "The question which is related to general knowledge, space or language.",
                    },
                },
                "required": ["question_for_topic"],
                "additionalProperties": False,
            },
        }

        self.ask_gemini_function = {
            "name": "ask_gemini",
            "description": (
                "Get the answer to the question related to a topic this agent is familiar with. Call this whenever "
                "you need to answer something related to sports, movies or games. Few examples: 'What is the difference "
                "between rugby and football?', 'What is WWE?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question_for_topic": {
                        "type": "string",
                        "description": "The question which is related to sports, movies or games.",
                    },
                },
                "required": ["question_for_topic"],
                "additionalProperties": False,
            },
        }

    def ask_assistant(self, history):
        """
        Handles interactions with the assistant by processing user inputs and generating responses.
        :param history: List of conversation history.
        :return: Updated conversation history.
        """
        prompts = [{"role": "system", "content": self.sys_prompt_gpt}] + history
        response = self.llm_manager.openai_client.chat.completions.create(
            model='gpt-4o-mini',
            tools=self.assistant_tools,
            messages=prompts
        )
        if response.choices[0].message.content is not None:
            prompts += [{"role": "assistant", "content":response.choices[0].message.content}]
        generated_content = response.choices[0].message.content or ""
        history += [{"role" : "assistant", "content" : ""}]
            
        if response.choices[0].finish_reason == "tool_calls":

            # Go through tool calls
            for tc in response.choices[0].message.tool_calls:
                tc_function_name = tc.function.name
                tc_function_args = json.loads(tc.function.arguments)
                tc_id = tc.id
            
                tool_response, tool_call = self.handle_tool_call(tc_function_name, tc_function_args, tc_id)
    
                prompts.append({
                    "role": "assistant",
                    "tool_calls": [tool_call]
                })
            
                prompts.append(tool_response)

            # Call the LLM again with updated context, but without the tools.
            updated_response = self.llm_manager.openai_client.chat.completions.create(
                                model='gpt-4o-mini',
                                messages=prompts)
            generated_content = updated_response.choices[0].message.content or ""
            # self.play_TTS(updated_response.choices[0].message.content)
        else:
            generated_content = response.choices[0].message.content or ""
            # self.play_TTS(response.choices[0].message.content)
        
        history[-1]['content'] = generated_content
        return history
    
    def ask_assistant_stream(self, history):
        """
        Handles interactions with the assistant by processing user inputs and generating responses in stream format.
        :param history: List of conversation history.
        :return: Updated conversation history in stream format.
        """
        prompts = [{"role": "system", "content": self.sys_prompt_gpt}] + history
        response = self.llm_manager.openai_client.chat.completions.create(
            model='gpt-4o-mini',
            tools=self.assistant_tools,
            messages=prompts,
            stream=True
        )
        generated_content = ""
        history += [{"role":"assistant", "content":generated_content}]

        tool_call_accumulator = ""  # Accumulator for JSON fragments of tool call arguments
        tool_call_id = None  # Current tool call ID
        tool_call_function_name = None # Function name
        tool_calls = []  # List to store complete tool calls
        
        for chunk in response:
            # Get Typewriting effect
            if chunk.choices[0].delta.content:
                generated_content += chunk.choices[0].delta.content or ""
                history[-1]['content'] = generated_content
                yield history
            
            # Check for tool_calls
            if chunk.choices[0].delta.tool_calls:
                for tc in chunk.choices[0].tool_calls:
                    if tc.id:   # New tool call detected here
                        tool_call_id = tc.id
                        if tool_call_function_name is None:
                            tool_call_function_name = tc.function.name
                    
                    tool_call_accumulator += tc.function.arguments if tc.function.arguments else ""
                    
                    # When the accumulated JSON string seems complete then:
                    try:
                        func_args = json.loads(tool_call_accumulator)
                        
                        # Handle tool call and get response
                        tool_response, tool_call = self.handle_tool_call(tool_call_function_name, func_args, tool_call_id)
                        tool_calls.append(tool_call)
                        
                        prompts.append({
                                    "role": "assistant",
                                    "tool_calls": tool_calls})
            
                        prompts.append(tool_response)
                        
                        # Call the LLM again with updated context, but without the tools.
                        updated_response = self.llm_manager.openai_client.chat.completions.create(
                                        model='gpt-4o-mini',
                                        messages=prompts,
                                        stream=True)
                        
                        # Reset and accumulate new full response
                        generated_content = ""
                        for chunk in updated_response:
                            if chunk.choices[0].delta.content:
                                generated_content += chunk.choices[0].delta.content or ""
                                history[-1]['content'] = generated_content
                                yield history
                        
                        # Reset tool call accumulator and related variables
                        tool_call_accumulator = ""
                        tool_call_id = None
                        tool_call_function_name = None
                        tool_calls = []

                    except json.JSONDecodeError as e:
                        print("Exception Occured : {e}")

        # self.play_TTS(generated_content)    
    
    def handle_tool_call(self, function_name, arguments, tool_call_id):
        """
        Handles tool calls invoked during assistant interactions.
        :param function_name: The name of the function being called.
        :param arguments: Arguments passed to the function.
        :param tool_call_id: Unique ID for the tool call.
        :return: Response from the tool and tool call metadata.
        """
        question = arguments.get('question_for_topic')
    
        # Prepare tool call information
        tool_call = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(arguments)
            }
        }
        
        if function_name in self.tools_functions_map:
            answer = self.tools_functions_map[function_name](question)
            response = {
                "role": "tool",
                "content": json.dumps({"question": question, "answer" : answer}),
                "tool_call_id": tool_call_id
            }

        return response, tool_call

    def ask_ollama(self, question):
        """
        Queries the Ollama model for general knowledge, space, or language-related topics.
        :param question: The input question.
        :return: Model response.
        """
        return self.llm_manager.query_llama(system_prompt=self.sys_prompt_ollama, msg_prompts=[{"role" : "user", "content": question}])
    
    def ask_gemini(self, question):
        """
        Queries the Gemini model for sports, movies, or games-related topics.
        :param question: The input question.
        :return: Model response.
        """
        return self.llm_manager.query_gemini(system_prompt=self.sys_prompt_gemini, msg_prompts=[{"role" : "user", "parts": [question]}])
    
    def play_audio_ffplay(self, audio_segment):
        """
        Plays an audio segment using ffplay.
        :param audio_segment: AudioSegment object to be played.
        """
        # Path for temporary audio file
        temp_file_path = os.path.join(self.temp_dir, "temp_output.wav")

        # Export audio to a temporary file
        audio_segment.export(temp_file_path, format="wav")

        try:
            # Play audio using ffplay
            subprocess.call(["ffplay", "-nodisp", "-autoexit", temp_file_path])
        finally:
            # Cleanup: Remove the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def play_TTS(self, message):
        """
        Converts text to speech and plays it.
        :param message: Text message to be converted to speech.
        """
        response = openai.audio.speech.create(model="tts-1", voice="onyx", input=message)
        audio_stream = BytesIO(response.content)
        audio = AudioSegment.from_file(audio_stream, format="mp3")
        self.play_audio_ffplay(audio)

    def transcribe_audio(self, audio_file_path):
        """
        Transcribes audio to text using Whisper.
        :param audio_file_path: Path to the audio file to be transcribed.
        :return: Transcribed text.
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
            gr.Info("Transcription Successful !", duration=5)
            return response.text
        except Exception as e:
            return f"An error occurred: {e}"

    def run(self):
        """
        Initializes and runs the Gradio UI.
        """
        with gr.Blocks() as ui:
            with gr.Row():
                chatbot = gr.Chatbot(height=600, type="messages", label="MultiModal Technical Expert Chatbot")
            with gr.Row():
                entry = gr.Textbox(label="Ask our technical expert anything:")
                audio_input = gr.Audio(
                    sources="microphone", 
                    type="filepath",
                    label="Record audio",
                    editable=False,
                    waveform_options=gr.WaveformOptions(
                        show_recording_waveform=False,
                    ),
                )

                # Add event listener for audio stop recording and show text on input area
                audio_input.stop_recording(
                    fn=self.transcribe_audio, 
                    inputs=audio_input, 
                    outputs=entry
                )        
            with gr.Row():
                clear = gr.Button("Clear")

            # Adds a new message in the history
            def do_entry(message, history):
                history += [{"role":"user", "content":message}]
                yield "", history
                
            entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry,chatbot]).then(self.ask_assistant_stream, inputs=chatbot, outputs=[chatbot])
            
            clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)
        ui.queue().launch()


if __name__ == "__main__":
    # Initialize and run the assistant
    assistant = MultiModal_Gradio_Assistant()
    assistant.run()
