# LC_RagBot

LC_RagBot is an advanced project that leverages Natural Language Processing (NLP) and state-of-the-art tools for document processing, embedding generation, and conversational AI. Built with a combination of Langchain, Gradio, Chroma, and other libraries, LC_RagBot is designed to streamline the process of creating a powerful retrieval-augmented chatbot.

## Features

- **Dynamic Document Loading**: Automatically loads documents from structured folders and assigns metadata to each document type.
- **Text Chunking**: Efficiently splits documents into manageable chunks for better processing and embedding.
- **Embeddings with Langchain**: Leverages advanced embedding techniques with OpenAI or HuggingFace for high-quality vector representation.
- **Conversational Retrieval**: Builds a conversational pipeline that retrieves contextually relevant information.
- **Visualizations**: Includes tools like Matplotlib and Plotly for creating insightful visualizations of data and embeddings.
- **Environment Configuration**: Manages environment variables using dotenv for seamless deployment.

## Project Structure

The project is organized into multiple stages for better readability and scalability:

1. **Imports and Configuration**: Handles library imports and environment variable loading.
2. **Document Processing**:
   - Reads and categorizes documents from the `knowledge-base` directory.
   - Adds metadata to documents for better categorization.
3. **Text Splitting**: Splits documents into smaller chunks using Langchain's `CharacterTextSplitter`.
4. **Embeddings**: Generates vector representations for the chunks using models like GPT-4 and HuggingFace embeddings.
5. **Retrieval Chain**: Implements a conversational retrieval chain to handle user queries.

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/brickster241/LLM_Projects.git
   cd LLM_Projects/LangChain-RAG-Bot
   ```

2. **Set Conda Environment using Anaconda :** 
   ```bash
   conda create --name projectenv python=3.11
   conda activate projectenv
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook:**
     ```bash
     jupyter lab
     ```

## 📦 File Structure

```plaintext
LLM-Projects/
├── .env                                            # .env File for storing OPENAI API Key
├── LangChain-RAG-Bot/                              # Project Directory
│   ├── knowledge-base                              # Knowledge Base (replace with your own !!)
│   ├── LC_RAGBot.ipynb                             # Jupyter Notebook file with RAGBot Implementation
│   ├── README.md                                   # Project documentation
│   ├── requirements.txt                            # Requirements.txt file for creating environment
```

