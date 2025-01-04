# LLM Summarizer

Welcome to the **LLM Summarizer**! This application utilizes cutting-edge Language Model (LLM) capabilities to generate concise and insightful summaries from diverse input formats. Whether you have a web article, a YouTube video, a PDF document, or an audio file, this tool can help distill key points efficiently.

---

## 🚀 Features

### 🌐 Summarize Website Content
- Simply provide a URL to any web page, and the tool will extract the main content and generate a summary.

### 📺 Summarize YouTube Videos
- Paste a YouTube video link to generate a text-based summary of its audio content (leveraging automated transcription).

### 📄 Summarize PDFs
- Upload a PDF document, and the tool will parse its text to create a concise summary.

### 🎧 Summarize Audio Files
- Upload an audio file, and the tool will convert its content to text using speech-to-text technology and generate a summary.

### 🌈 Seamless User Experience
- Built using **Streamlit**, the web interface is simple, responsive, and user-friendly.

---

## 🛠️ How It Works

1. **Input Formats:**
   - URL (Website article)
   - YouTube Video Link
   - PDF File
   - Audio File

2. **Processing:**
   - Extracts relevant content from the input.
   - Uses an LLM (e.g., OpenAI's GPT) to analyze and summarize the content.

3. **Output:**
   - Displays the summary directly on the web page.
   - Option to download the summary as a text file.

---

## 🧑‍💻 Technologies Used

- **Backend:** OpenAI GPT-4o-mini (or another LLM API).
- **Frontend:** Streamlit for a dynamic web interface.
- **Additional Tools:**
  - `BeautifulSoup` or `Selenium` for web scraping.
  - `PyPDF2` or `pdfplumber` for PDF parsing.
  - `yt-dlp` and `whisper` for YouTube transcription.
  - `Whisper` or other speech-to-text tools for audio file processing.

---

## 🖥️ Installation and Usage

### Prerequisites
- Python 3.11+
- An OpenAI API Key (or key for your chosen LLM provider).

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

3. Enter a URL, YouTube video link, upload a PDF file, or upload an audio file to get your summary!

---

## 📦 File Structure

```plaintext
LLM-Projects/
├── .env             # .env File for storing OPENAI API Key
├── environment.yml   # List of dependencies
├── README.md          # Project documentation
├── LLM-Summarizer/             # Project Directory
│   ├── pdf_handler.py  # PDF parsing logic
│   ├── web_scraper.py  # Web scraping logic
│   ├── yt_transcriber.py # YouTube transcription logic
│   ├── audio_handler.py  # Audio file processing logic
│   └── app.py          # Main Streamlit App
```

---

## 🌟 Example

1. Enter a **web page URL**, and you'll receive:
   - A summary of the key points from the article.

2. Paste a **YouTube link**, and get:
   - A text summary of the video's content (audio transcription included).

3. Upload a **PDF file**, and:
   - A concise summary of its content is displayed.

4. Upload an **audio file**, and:
   - A text transcription of its content is generated along with a summary.
