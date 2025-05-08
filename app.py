import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables (OPENAI_API_KEY)
load_dotenv()


def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text


def get_text_chunks(text, chunk_size=2000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def build_vector_store(text_chunks, index_dir="faiss_index"):
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(text_chunks, embeddings)
    store.save_local(index_dir)
    return store


def load_vector_store(index_dir="faiss_index"):
    embeddings = OpenAIEmbeddings()
    # Enable deserialization of FAISS index you created
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_qa_chain(model_name="gpt-3.5-turbo", temperature=0.2):
    prompt_template = """
Answer the question as detailed as possible from the provided context. If the answer is not in the context, reply "Answer not available in the context.".

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    model = ChatOpenAI(model_name=model_name, temperature=temperature)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def answer_query(question, vector_store):
    docs = vector_store.similarity_search(question)
    chain = get_qa_chain()
    result = chain.run(input_documents=docs, question=question)
    return result


def main():
    st.set_page_config(page_title="GPT-PDF Chatbot", page_icon=":books:")
    st.header("📚 GPT-PDF Chatbot 🤖")

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_files = st.file_uploader(
            "Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Process PDFs"):
            if pdf_files:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_files)
                    chunks = get_text_chunks(raw_text)
                    build_vector_store(chunks)
                    st.success("PDFs processed and index created!")
            else:
                st.warning("Please upload at least one PDF.")

    # Main input box
    user_question = st.text_input("Ask a question about your PDFs:")
    if user_question:
        try:
            vector_store = load_vector_store()
            with st.spinner("Finding answer..."):
                answer = answer_query(user_question, vector_store)
            st.write("**Answer:**")
            st.write(answer)
        except Exception as e:
            st.error(f"Error loading index or retrieving answer: {e}")

    # Footer
    st.markdown(
        "---\n"
        "Made with ❤️ by [Vrinda Rani](https://github.com/vrindarani00)"
    )

if __name__ == "__main__":
    main()