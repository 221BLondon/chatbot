import streamlit as st
import pymupdf4llm
import time
import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Initialize HuggingFace Embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'})
vector_store = Chroma(embedding_function=embeddings)

# Streamlit UI
st.title("📚 RAG Chatbot - PDF Based Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        file_path = f"./{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Convert PDF to Markdown
        docs = pymupdf4llm.to_markdown(file_path)
        markdown_path = file_path.replace(".pdf", ".md")
        with open(markdown_path, "w", encoding="utf-8") as md_file:
            md_file.write(docs)
        
        # Load markdown content
        loader = UnstructuredMarkdownLoader(markdown_path)
        docs = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)
        
        # Store embeddings in vector database
        for i in range(0, len(all_splits), 10):  # Batch processing
            batch = all_splits[i : i + 10]
            vector_store.add_documents(documents=batch)
            time.sleep(2)
        
        st.success("PDF processed and stored successfully! You can now ask questions.")

# Chatbot Interface
st.header("💬 Ask a Question")
question = st.text_input("Enter your question:")

groq_api_key = st.text_input("Enter Groq API Key:", type="password")
if question and groq_api_key:
    model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0)
    
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Generate response
    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks. Use the following retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Keep the answer concise.
        {context}
        Question: {question}
        Helpful Answer:
        """
    )
    
    chain = prompt | model
    response = chain.invoke({"context": context, "question": question}).content
    
    st.subheader("🤖 AI Response")
    st.write(response)
