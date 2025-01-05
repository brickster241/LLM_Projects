# LLM Summarizer

Welcome to the **LLM Summarizer**! This application utilizes cutting-edge Language Model (LLM) capabilities to generate concise and insightful summaries from diverse input formats. Whether you have a web article, a YouTube video, a PDF document, or an audio file, this tool can help distill key points efficiently.

---

## 🚀 Features

### 📺 Summarize YouTube Videos
- Upload a custom .mp4 video to generate a text-based summary of its audio content (leveraging automated transcription).

### 📄 Summarize PDFs
- Upload a PDF document, and the tool will parse its text to create a concise summary.

### 🎧 Summarize Audio Files
- Upload an audio file, and the tool will convert its content to text using speech-to-text technology and generate a summary.

### ✂️ Character Limit Control
- Summaries are tailored to fit your desired length, ensuring they remain concise and relevant.

### 🌈 Seamless User Experience
- Built using **Streamlit**, the web interface is simple, responsive, and user-friendly.

---

## 🛠️ How It Works

1. **Input Formats:**
   - .mp4 Video File
   - PDF File
   - .mp3 / .wav Audio File

2. **Processing:**
   - Extracts relevant content from the input.
   - Uses an LLM (e.g., OpenAI's GPT-4o-mini) to analyze and summarize the content.

3. **Output:**
   - Displays the summary directly on the web page.

---

## 🧑‍💻 Technologies Used

- **Backend:** OpenAI GPT-4o-mini.
- **Frontend:** Streamlit for a dynamic web interface.
- **Additional Tools:**
  - `PyPDF2` for PDF parsing.
  - `Whisper` for speech-to-text tools and audio/video file processing.

---

## 🖥️ Installation and Usage

### Prerequisites
- Python 3.11+
- An OpenAI API Key.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/brickster241/LLM_Projects.git
   ```

2. Create environment and install required dependencies using Anaconda:
   ```bash
   conda env create -f environment.yml
   ```

3. Add a `.env` file, containing `OPEN_API_KEY=yourAPIKey`.

### Usage

1. Navigate to the LLM-Summarizer directory and Run the Streamlit app:
   ```bash
   cd LLM_Summarizer
   streamlit run app.py
   ```

2. Open your browser and navigate to the local URL provided by Streamlit (e.g., `http://localhost:8501`).

3. Upload video, upload a PDF file, or upload an audio file to get your summary!

---

## 📦 File Structure

```plaintext
LLM-Projects/
├── .env                         # .env File for storing OPENAI API Key
├── environment.yml              # List of dependencies
├── LLM-Summarizer/              # Project Directory
│   ├── llm_summarizer.py        # PDF file, Audio file, Youtube URL processing logic
│   ├── streamlit_app.py         # Main Streamlit App
│   ├── README.md                # Project documentation
```

---