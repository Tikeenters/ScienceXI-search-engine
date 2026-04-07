import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
import os
import glob

# 1. Page Setup (Clean and Wide)
st.set_page_config(page_title="Academic Revenge Engine", layout="wide")
st.title("Study Search ")
st.markdown("Your AI-powered study buddy for Class 10. Currently science!")

# 2. Secure API Key
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("API Key not found. Please set it in Streamlit Secrets.")
    st.stop()

# 3. Memory Initialization
if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

# 4. The Brain Configuration (with Easy Mode)
def get_conversational_chain(easy_mode):
    if easy_mode:
        prompt_template = """
        You are a friendly science tutor. Explain the answer simply, as if you were talking to a beginner.
        Use easy words, short sentences, and everyday examples without lengthening the answer. Do not use overly complex jargon.
        If the answer is not in the provided context just say, "The answer is not available in the textbook."\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """
    else:
        prompt_template = """
        Answer the question as detailed as possible from the provided context. Include all technical terms. 
        If the answer is not in the provided context just say, "The answer is not available in the textbook."\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """

    # Using the stable, current model names
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# 5. THE NEW GEMINI-STYLE LAYOUT
# We use vertical_alignment="bottom" so the buttons line up perfectly with the text box
col_left, col_center, col_right = st.columns([1.5, 6, 1.5], vertical_alignment="bottom")

# LEFT MENU: The Chapter Tools
with col_left:
    with st.popover("📚 Chapters Tools"):
        st.markdown("**Load AI Brain**")
        st.caption("Select up to 5 chapters to study.")
        
        pdf_files = glob.glob("*.pdf")
        selected_chapters = st.multiselect(
            "Select PDFs:",
            options=pdf_files,
            label_visibility="collapsed" # Hides the label to keep it clean
        )
        
        if st.button("Load Brain", use_container_width=True):
            if len(selected_chapters) == 0:
                st.error("Select a chapter first!")
            elif len(selected_chapters) > 5:
                st.error("Max 5 chapters allowed.")
            else:
                with st.spinner("Reading pages..."):
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
                                    continue # Diagram shock absorber
                        except Exception as e:
                            st.warning(f"Skipped {pdf_path}: {e}")
                            continue
                    
                    if text:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                        chunks = text_splitter.split_text(text)
                        
                        try:
                            embeddings = GoogleGenerativeAIEmbeddings(
                                model="models/gemini-embedding-001",
                                task_type="retrieval_document"
                            )
                            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                            vector_store.save_local("faiss_index")
                            st.session_state.is_ready = True
                            st.toast("🧠 AI Brain Loaded Successfully!") # A sleek pop-up notification
                        except Exception as e:
                            st.error(f"Google API Error: {e}")
                    else:
                        st.error("No readable text found.")

# RIGHT MENU: The Easy Toggle
with col_right:
    is_easy_mode = st.toggle("Easy Mode", value=False)

# CENTER MENU: The Main Search Bar
with col_center:
    user_question = st.text_input(
        "Search:", 
        placeholder="Type your question and press Enter...", 
        label_visibility="collapsed"
    )

# 6. The Answer Processing
if user_question:
    if not st.session_state.is_ready:
        st.warning("👈 Please open the Chapters Tools and load your PDFs first!")
    else:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                task_type="retrieval_document"
            )
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            
            chain = get_conversational_chain(is_easy_mode)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            
            st.markdown("---")
            st.markdown("### Answer:")
            st.write(response["output_text"])
        except Exception as e:
            st.error(f"Search failed: {e}")
