import streamlit as st
import os
import uuid
import pickle
import shutil
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import requests
from streamlit_lottie import st_lottie
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=7, memory_key="chat_history", return_messages=True)
if "editing_message_index" not in st.session_state:
    st.session_state.editing_message_index = None
if "show_animation" not in st.session_state:
    st.session_state.show_animation = True

# Set up medical-themed CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f7ff;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput > div > div > input {
        border: 2px solid #007bff;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #0056b3;
    }
    .sidebar .sidebar-content {
        background-color: #e6f0ff;
    }
    .chat-message-user {
        background-color: #007bff;
        color: white;
        border-radius: 15px;
        padding: 10px;
        margin: 5px;
        max-width: 70%;
        float: right;
        clear: both;
    }
    .chat-message-assistant {
        background-color: #e9ecef;
        color: black;
        border-radius: 15px;
        padding: 10px;
        margin: 5px;
        max-width: 70%;
        float: left;
        clear: both;
    }
    .stFileUploader > div > div {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Function to load Lottie animation
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Display medical-themed animation on startup
if st.session_state.show_animation:
    lottie_medical = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_1pxqjqgy.json")
    st_lottie(lottie_medical, height=300, key="medical_animation")
    st.markdown("<h1 style='text-align: center; color: #007bff;'>MediChat: Your AI Medical Assistant</h1>", unsafe_allow_html=True)
    st.session_state.show_animation = False

# Load and split PDF data
def load_pdf_file(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download HuggingFace Embeddings
def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Set up Groq LLM
def setup_llm():
    return ChatGroq(
        temperature=0.4,
        max_tokens=500,
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

# Create FAISS vector store
def create_faiss_vector_store(documents, embeddings, store_id):
    vector_store_path = f"faiss_store_{store_id}"
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(vector_store_path)
    return vector_store_path

# Load FAISS vector store
def load_faiss_vector_store(store_id, embeddings):
    vector_store_path = f"faiss_store_{store_id}"
    if os.path.exists(vector_store_path):
        return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return None

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Medical Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        # Create a unique ID for the new vector store
        new_store_id = str(uuid.uuid4())
        temp_dir = f"temp_{new_store_id}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded PDF
        pdf_path = os.path.join(temp_dir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Process PDF
        extracted_data = load_pdf_file(temp_dir)
        text_chunks = text_split(extracted_data)
        embeddings = download_hugging_face_embeddings()
        vector_store_path = create_faiss_vector_store(text_chunks, embeddings, new_store_id)
        
        # Update session state
        st.session_state.vector_store_id = new_store_id
        st.session_state.chat_history = []  # Clear chat history for new document
        st.session_state.memory = ConversationBufferWindowMemory(k=7, memory_key="chat_history", return_messages=True)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        
        st.success("PDF processed successfully! Start chatting.")

# Main chat interface
st.header("Chat with MediChat")
if st.session_state.vector_store_id:
    # Set up RAG chain
    embeddings = download_hugging_face_embeddings()
    vector_store = load_faiss_vector_store(st.session_state.vector_store_id, embeddings)
    if vector_store:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        llm = setup_llm()
        
        system_prompt = (
            "You are a medical assistant for question-answering tasks. "
            "Use the provided medical documents to answer the question accurately. "
            "If you don't know the answer, say so. Keep answers concise, up to three sentences.\n\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Display chat history
        for idx, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                with st.container():
                    st.markdown(f'<div class="chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
                    if st.button("Edit", key=f"edit_{idx}"):
                        st.session_state.editing_message_index = idx
            else:
                with st.container():
                    st.markdown(f'<div class="chat-message-assistant">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Handle message editing
        if st.session_state.editing_message_index is not None:
            with st.form(key="edit_form"):
                edited_message = st.text_input("Edit your message:", value=st.session_state.chat_history[st.session_state.editing_message_index]["content"])
                submit_edit = st.form_submit_button("Submit Edit")
                if submit_edit:
                    # Update chat history
                    st.session_state.chat_history[st.session_state.editing_message_index]["content"] = edited_message
                    # Reprocess the edited query
                    response = rag_chain.invoke({"input": edited_message, "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]})
                    # Update chat history with new response
                    st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
                    st.session_state.memory.save_context({"input": edited_message}, {"output": response["answer"]})
                    st.session_state.editing_message_index = None
                    st.rerun()
        
        # Chat input
        user_input = st.text_input("Ask a medical question:", key="user_input")
        if user_input:
            # Invoke RAG chain
            response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]})
            # Update chat history and memory
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
            st.session_state.memory.save_context({"input": user_input}, {"output": response["answer"]})
            st.rerun()
    else:
        st.error("Failed to load vector store. Please upload a new PDF.")
else:
    st.info("Please upload a PDF to start chatting.")
