import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
import os
import glob

# 1. Page Setup
st.set_page_config(page_title="ScienceX Search Engine", layout="wide")
st.header("ScienceX Search Engine 🔬")

# 2. Get the API Key securely from Streamlit
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Key not found. Please set it in Streamlit Secrets to resurrect the AI.")
    st.stop()

# Initialize memory so the app doesn't forget the database exists after a button click
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

# 3. SIDEBAR: The Chapter Sorter
st.sidebar.title("📚 Study Material")
st.sidebar.warning("⚠️ **Pro-Tip:** Google's free tier gets violently angry if you upload too much text at once. Select 5 chapters or fewer.")

pdf_files = glob.glob("*.pdf")
selected_chapters = st.sidebar.multiselect(
    "Choose chapters to load into the AI Brain:",
    options=pdf_files,
    help="Select the chapters related to your question."
)

# 4. Background Setup: Triggered only when you click the button
if st.sidebar.button("Load AI Brain"):
    if len(selected_chapters) == 0:
        st.sidebar.error("You have to give it a brain to work with. Select a chapter.")
    elif len(selected_chapters) > 5:
        st.sidebar.error("Too many chapters! Please select 5 or fewer.")
    else:
        with st.spinner(f"Force-feeding {len(selected_chapters)} chapters into the AI..."):
            text = ""
            for pdf_path in selected_chapters:
                try:
                    pdf_reader = PdfReader(pdf_path)
                    for page in pdf_reader.pages:
                        try:
                            extracted = page.extract_text()
                            if extracted:
                                text += extracted
                        except Exception:
                            continue # The shock absorber for messy diagrams
                except Exception as e:
                    st.sidebar.warning(f"Could not read {pdf_path}: {e}")
                    continue
            
            if text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                chunks = text_splitter.split_text(text)
                
                try:
                    # Explicit task_type keeps the API from rejecting the payload
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-004",
                        task_type="retrieval_document"
                    )
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    vector_store.save_local("faiss_index")
                    st.session_state.is_ready = True
                    st.sidebar.success("Brain Loaded! You can now ask questions.")
                except Exception as e:
                    st.sidebar.error(f"Google AI Error: {e}")
            else:
                st.sidebar.error("Failed to extract any readable text from those PDFs.")

# 5. Function to set up the AI Chat parameters
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

# 6. The User Interface (Only shows up if the Brain is loaded)
if st.session_state.is_ready:
    user_question = st.text_input("Ask a question about your loaded chapters:")
    if user_question:
        try:
            # We must use the exact same embedding configuration to read the database as we used to write it
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-004",
                task_type="retrieval_document"
            )
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
            st.markdown("### Answer:")
            st.write(response["output_text"])
        except Exception as e:
            st.error(f"Error searching the database: {e}")
else:
    st.info("👈 Please select and load your chapters from the sidebar to begin.")
