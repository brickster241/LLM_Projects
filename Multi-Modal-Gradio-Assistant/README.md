# Multi-Modal-Gradio-Assistant
## Overview
The **Multi-Modal Gradio Assistant** is a versatile AI assistant that leverages the power of large language models (LLMs) to deliver expert-level responses across multiple domains. The assistant supports both text and audio input and provides responses in text and synthesized speech. It can also utilize external tools to enhance its capabilities, making it a comprehensive solution for multimodal AI interactions.

---

## Features

### Multimodal Interaction
- **Text Input and Output**: Users can interact with the assistant through a text-based interface.
- **Audio Input and Output**: Users can provide voice input, which is transcribed into text, and the assistant can respond using text-to-speech (TTS).

### Domain Expertise
- **Technical Topics**: Expertise in coding, software engineering, data science, and LLMs.
- **Specialized Topics**: Sports, movies, games, space, and general knowledge.

### Tool Integration
- **Ask Gemini**: Handles queries related to sports, movies, and games.
- **Ask Ollama**: Answers general knowledge questions, including those related to space and spoken languages.

### Additional Capabilities
- **Streaming Responses**: Provides responses incrementally for real-time interaction.
- **Speech Synthesis**: Reads out responses after streaming.
- **Voice Transcription**: Converts voice input into text for seamless interaction.

---

## Architecture

### Core Components
1. **LLM Manager**: Handles interactions with various language models.
2. **Temporary Audio Directory**: Manages temporary audio files for TTS playback.
3. **System Prompts**: Defines role-based prompts for different domains.
4. **Tool Mapping**: Maps tools (e.g., Ask Gemini, Ask Ollama) to specific functionalities.

### Supported Models
- OpenAI `Whisper` for audio transcription.
- OpenAI `gpt-4o-mini`, Google `gemini-1.5-flash` and Ollama `llama3.2:3b` for text generation.
- OpenAI TTS for speech synthesis.

---

## Installation

### Prerequisites
- Python 3.11+
- An OpenAI API Key, Anthropic API Key and Google API Key.
- Ollama's Open source Models (`llama3.2:3b`).
- Ensure you have ffmpeg installed and added to PATH, for Speech-to-Text conversion.

### Steps
1. **Clone this repository:**
   ```bash
   git clone https://github.com/brickster241/LLM_Projects.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd LLM_Projects/Multi-Modal-Gradio-Assistant
   ```
3. **Create environment using conda and install dependencies:**
   ```bash
   conda create --name gradio_bot_env python=3.11
   conda activate gradio_bot_env
   pip install -r requirements.txt
   ```
4. **Add a `.env` file based on file structure below, containing all your API Keys :** 
   ```code
   OPENAI_API_KEY=yourOPENAIAPIKey
   GOOGLE_API_KEY=yourGOOGLEAPIKey
   ```
5. **Run the assistant:**
   ```bash
   python multimodal_gradio_assistant.py
   ```

---

## Usage

1. Launch the assistant by running the script.
2. Interact using the Gradio interface:
   - Type your queries in the text box.
   - Record audio input using the microphone option.
3. View responses in the chat interface or listen to them via TTS.

---

## 📦 File Structure

```plaintext
LLM-Projects/
├── .env                                            # .env File for storing OPENAI, GOOGLE API Key
├── Multi-Modal-Gradio-Assistant/                   # Project Directory
│   ├── llm_manager.py                              # class to handle different LLM connections, conversations
│   ├── multimodal_gradio_assistant.py              # Gradio Chatbot App
│   ├── README.md                                   # Project documentation
│   ├── requirements.txt                            # Requirements.txt file for creating environment
```

---

## Code Highlights

### Assistant Initialization
- Ensures a temporary directory for audio files exists.
- Initializes the LLM manager and tools.

```python
self.temp_dir = os.path.join(os.path.dirname(__file__), "TempAudio")
os.makedirs(self.temp_dir, exist_ok=True)
self.llm_manager = LLM_Manager()
```

### Tool Handling
- Tools like **Ask Gemini** and **Ask Ollama** are dynamically invoked based on user queries.

```python
self.tools_functions_map = {
    "ask_ollama": self.ask_ollama,
    "ask_gemini": self.ask_gemini,
}
```

### Multimodal Input and Output
- Uses Whisper for transcription and OpenAI TTS for speech synthesis.

```python
def transcribe_audio(self, audio_file_path):
    response = openai.audio.transcriptions.create(
        model="whisper-1", file=audio_file
    )
    return response.text

def play_TTS(self, message):
    response = openai.audio.speech.create(model="tts-1", voice="onyx", input=message)
```

---

## Gradio Interface

- **Chatbot**: Displays the conversation.
- **Textbox**: Accepts user queries.
- **Audio Input**: Records voice queries and transcribes them.

```python
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=600, type="messages", label="MultiModal Technical Expert Chatbot")
    with gr.Row():
        entry = gr.Textbox(label="Ask our technical expert anything:")
        audio_input = gr.Audio(sources="microphone", type="filepath", label="Record audio")
```
---
## **Future Plans**

- Add Image generation as a tool by using `DALL-E-3` Model.
- Support more open-source LLMs in specialized domains.

---