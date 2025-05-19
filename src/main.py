import streamlit as st
import os
import uuid
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

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

# Display title and instructions
st.markdown("<h1 style='text-align: center; color: #007bff;'>MediChat: Your AI Medical Assistant</h1>", unsafe_allow_html=True)
st.markdown("Upload a medical document (PDF) to ask questions about its content. MediChat will provide concise, accurate answers based on the document.", unsafe_allow_html=True)

# Load and split PDF data
def load_pdf_file(data_path):
    try:
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {data_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        st.error(f"Failed to load PDF: {str(e)}")
        return []

def text_split(extracted_data):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(extracted_data)
        logger.info(f"Split into {len(text_chunks)} text chunks")
        return text_chunks
    except Exception as e:
        logger.error(f"Error splitting text: {str(e)}")
        st.error(f"Failed to split text: {str(e)}")
        return []

# Check if PDF content is medical-related
def is_medical_content(documents):
    medical_keywords = ["medical", "health", "disease", "treatment", "symptoms", "diagnosis", "patient", "medicine", "hospital", "therapy"]
    for doc in documents:
        content = doc.page_content.lower()
        if any(keyword in content for keyword in medical_keywords):
            return True
    return False

# Cache embeddings to avoid reinitialization
@st.cache_resource
def download_fastembed_embeddings():
    try:
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        logger.info("Initialized FastEmbed embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing FastEmbed embeddings: {str(e)}")
        st.error(f"Failed to initialize embeddings: {str(e)}")
        return None

# Set up Groq LLM with network error handling
def setup_llm():
    try:
        llm = ChatGroq(
            temperature=0.4,
            max_tokens=500,
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        logger.info("Initialized Groq LLM")
        return llm
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error initializing Groq LLM: {str(e)}")
        st.error("Failed to connect to Groq API. Please check your internet connection or API key.")
        return None
    except Exception as e:
        logger.error(f"Error initializing Groq LLM: {str(e)}")
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

# Create FAISS vector store
def create_faiss_vector_store(documents, embeddings, store_id):
    try:
        vector_store_path = f"faiss_store_{store_id}"
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(vector_store_path)
        logger.info(f"Created FAISS vector store at {vector_store_path}")
        return vector_store_path
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {str(e)}")
        st.error(f"Failed to create vector store: {str(e)}")
        return None

# Cache vector store to avoid reloading
@st.cache_resource
def load_faiss_vector_store(store_id, _embeddings):
    try:
        vector_store_path = f"faiss_store_{store_id}"
        if os.path.exists(vector_store_path):
            vector_store = FAISS.load_local(vector_store_path, _embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded FAISS vector store from {vector_store_path}")
            return vector_store
        else:
            logger.error(f"Vector store path {vector_store_path} does not exist")
            st.error("Vector store not found. Please upload a new PDF.")
            return None
    except Exception as e:
        logger.error(f"Error loading FAISS vector store: {str(e)}")
        st.error(f"Failed to load vector store: {str(e)}")
        return None

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Medical Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
    if st.button("Reset and Upload New PDF"):
        st.session_state.clear()
        st.rerun()
    if uploaded_file and not st.session_state.pdf_processed:
        try:
            # Create a unique ID for the new vector store
            new_store_id = str(uuid.uuid4())
            temp_dir = f"pdf_storage/{new_store_id}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save uploaded PDF persistently
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process PDF
            extracted_data = load_pdf_file(temp_dir)
            if not extracted_data:
                st.error("No documents loaded from PDF. Please upload a valid PDF.")
                st.session_state.pdf_processed = False
                st.rerun()
            
            # Check if content is medical-related
            if not is_medical_content(extracted_data):
                st.warning("The uploaded PDF does not appear to contain medical content. Please upload a medical document for best results.")
            
            text_chunks = text_split(extracted_data)
            if not text_chunks:
                st.error("No text chunks created from PDF. Ensure the PDF contains extractable text.")
                st.session_state.pdf_processed = False
                st.rerun()
            
            embeddings = download_fastembed_embeddings()
            if not embeddings:
                st.error("Failed to initialize embeddings.")
                st.session_state.pdf_processed = False
                st.rerun()
            
            vector_store_path = create_faiss_vector_store(text_chunks, embeddings, new_store_id)
            if not vector_store_path:
                st.error("Failed to create vector store.")
                st.session_state.pdf_processed = False
                st.rerun()
            
            # Update session state
            st.session_state.vector_store_id = new_store_id
            st.session_state.chat_history = []
            st.session_state.memory = ConversationBufferWindowMemory(k=7, memory_key="chat_history", return_messages=True)
            st.session_state.pdf_processed = True
            
            st.success("PDF processed successfully! Start chatting.")
        except Exception as e:
            logger.error(f"Error processing PDF upload: {str(e)}")
            st.error(f"Error processing PDF: {str(e)}")
            st.session_state.pdf_processed = False

# Main chat interface
st.header("Chat with MediChat")
logger.info(f"Rendering chat UI with vector_store_id: {st.session_state.vector_store_id}")
if st.session_state.vector_store_id and st.session_state.pdf_processed:
    # Set up RAG chain
    embeddings = download_fastembed_embeddings()
    if not embeddings:
        st.error("Failed to initialize embeddings for chat.")
        st.stop()
    
    vector_store = load_faiss_vector_store(st.session_state.vector_store_id, embeddings)
    if vector_store:
        try:
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            llm = setup_llm()
            if not llm:
                st.error("Failed to initialize LLM for chat.")
                st.stop()
            
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
                        st.session_state.chat_history[st.session_state.editing_message_index]["content"] = edited_message
                        response = rag_chain.invoke({"input": edited_message, "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]})
                        logger.info(f"Processed edited query: {edited_message}")
                        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
                        st.session_state.memory.save_context({"input": edited_message}, {"output": response["answer"]})
                        st.session_state.editing_message_index = None
                        st.rerun()
            
            # Chat input with form to prevent repeated submissions
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input("Ask a medical question (e.g., 'What are the symptoms of diabetes?'):", key="user_input")
                submit_button = st.form_submit_button("Submit")
                if submit_button and user_input:
                    try:
                        response = rag_chain.invoke({"input": user_input, "chat_history": st.session_state.memory.load_memory_variables({})["chat_history"]})
                        logger.info(f"Processed user query: {user_input}")
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
                        st.session_state.memory.save_context({"input": user_input}, {"output": response["answer"]})
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error processing user input: {str(e)}")
                        st.error(f"Error processing your question: {str(e)}")
        except Exception as e:
            logger.error(f"Error setting up RAG chain: {str(e)}")
            st.error(f"Failed to set up chat system: {str(e)}")
    else:
        st.error("Failed to load vector store. Please upload a new PDF.")
        st.session_state.pdf_processed = False
else:
    st.info("Please upload a PDF to start chatting.")
