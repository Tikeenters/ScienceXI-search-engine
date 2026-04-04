import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
import glob # This new tool helps us find multiple files

# 1. Page Setup
st.set_page_config(page_title="ScienceX Search Engine", layout="wide")
st.header("ScienceX Search Engine 🔬")

# 2. Get the API Key securely from Streamlit
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.warning("API Key not found. Please set it in Streamlit Secrets.")

# 3. Background Setup: Read ALL PDFs automatically (Runs only once)
@st.cache_resource
def initialize_database():
    # Find all files in the folder that end in .pdf
    pdf_files = glob.glob("*.pdf") 
    
    if len(pdf_files) == 0:
        st.error("Could not find any PDF files in the project folder!")
        return False

    # Extract text from EVERY pdf found
    text = ""
    for pdf_path in pdf_files:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
        
    # Chop all the combined text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    # Create searchable database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return True

# 4. Show a loading spinner while the background setup runs
with st.spinner(f"Initializing AI Brain with all textbook chapters. This takes a minute on first load..."):
    is_ready = initialize_database()

# 5. Function to set up the AI Brain (Gemini Flash)
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context just say, "The answer is not available in the textbook", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# 6. The User Interface (Only shows up if the PDFs were successfully read)
if is_ready:
    user_question = st.text_input("Ask a question about the Class 11 Science textbook:")
    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("Reply: ", response["output_text"])
