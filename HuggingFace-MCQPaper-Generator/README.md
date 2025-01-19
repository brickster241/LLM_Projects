# LLM-Based MCQ Paper Generator

Generate customized multiple-choice question (MCQ) papers using a Hugging Face language model. This application allows users to input a domain and select a difficulty level to create well-structured MCQ papers.

---

## 🚀 Features

- **Domain-Specific MCQs**: Generate questions tailored to fields like Mathematics, Physics, Coding, etc.
- **Difficulty Levels**: Choose from Student, Graduate, Professional, or Expert levels.
- **Formatted Output**: Outputs the MCQ paper in Markdown format for easy sharing or export.
- **Interactive UI**: User-friendly Streamlit interface for seamless operation.

---

## 🏗️ Architecture

### **HF_MCQGenerator Class**
Encapsulates the logic for:
1. **Login**: Connects to Hugging Face Hub.
2. **Model Initialization**: Loads a pre-trained language model (`meta-llama/Llama-3.2-3B-Instruct`) with quantization for efficient inference.
3. **Tokenizer Setup**: Configures the tokenizer for text processing.
4. **MCQ Generation**: Generates papers based on the given domain and difficulty.

### **Streamlit App**
- **Sidebar**:
  - Title and Description with instructions.
- **Input Fields**:
  - Domain (Text Field).
  - Difficulty (Dropdown).
- **Output Section**:
  - Displays the generated MCQ paper in Markdown format.

---

## Installation

### Steps
1. **Clone this repository:**
   ```bash
   git clone https://github.com/brickster241/LLM_Projects.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd LLM_Projects/HuggigFace-MCQPaper-Generator
   ```
3. **Create environment using conda and install dependencies:**
   ```bash
   conda create --name hf_mcq_env python=3.11
   conda activate hf_mcq_env
   pip install -r requirements.txt
   ```
4. Add a `.env` file according to File Structure below, containing Huggingface Token : 
   ```code
   HF_TOKEN=yourHuggingFaceToken
   ```
5. **Run the App**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📦 File Structure

```plaintext
LLM-Projects/
├── .env                                            # .env File for storing HF_TOKEN Key
├── HuggingFace-MCQPaper-Generator/                 # Project Directory
│   ├── streamlit_app.py                            # Streamlit Application
│   ├── hf_mcq_generator.py                         # class which generates MCQ papers using a Hugging Face LM.
│   ├── README.md                                   # Project documentation
│   ├── requirements.txt                            # Requirements.txt file for creating environment
```

---

## ⚠️ Requirements & Limitations

- **Hardware Requirements**:
  - GPU with at least **6 GB RAM**.
  - Alternatively, run the app on **Google Colab** with T4 GPU as hosted runtime enabled.
- **Model Loading Time**: Initial setup may take a few minutes as the model and tokenizer are downloaded.

---
