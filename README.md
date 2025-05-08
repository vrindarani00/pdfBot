
# PDFbot

A simple Streamlit app that lets you upload PDF files, build a FAISS vector index over their text using OpenAI embeddings, and ask questions about the content via a chat interface.

---

## Features

- Upload one or more PDF files
- Extract text and split into overlapping chunks
- Build and save a FAISS index locally
- Load the saved index (with safe deserialization)
- Ask natural-language questions and get detailed answers from your PDFs

---

## Requirements

- Python 3.7+
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [LangChain](https://pypi.org/project/langchain/)
- [faiss-cpu](https://pypi.org/project/faiss-cpu/)
- [OpenAI Python client](https://pypi.org/project/openai/)

---

## Setup

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/gpt-pdf-chatbot.git
   cd gpt-pdf-chatbot
````

2. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   * Copy `.env.example` (or create a new `.env`) and add your OpenAI API key:

     ```env
     OPENAI_API_KEY=your_api_key_here
     ```

---

## Usage

1. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

2. **In your browser**

   * Open the sidebar and upload one or more PDF files
   * Click **Process PDFs** to extract text and build the FAISS index
   * Enter a question in the main input box to query your PDFs

---

## Project Structure

```
.
├── app.py            # Main Streamlit application
├── requirements.txt  # Python dependencies
├── .env.example      # Example environment file
└── faiss_index/      # Saved FAISS index directory (created at runtime)
```
