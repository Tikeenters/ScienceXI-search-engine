🔬 Science X Search Engine

An AI-powered search engine designed to index and query Class 10 (Science X) NCERT textbooks. Built using Python, Streamlit, and Google Gemini 1.5 Flash.
🚀 Overview

This project allows students and teachers to search through multiple science textbook chapters simultaneously. Instead of flipping through hundreds of pages, users can ask natural language questions and get precise answers based strictly on the textbook content.
🛠️ Tech Stack

    Frontend: Streamlit (Web Interface)

    LLM: Google Gemini 1.5 Flash (Small Language Model)

    Orchestration: LangChain (RAG Pipeline)

    Vector Database: FAISS (Efficient Similarity Search)

    PDF Processing: PyPDF2

📖 How It Works (RAG Architecture)

The application follows the Retrieval-Augmented Generation (RAG) workflow:

    Ingestion: The app automatically reads all .pdf files (NCERT chapters) in the repository.

    Chunking: The text is broken into smaller, overlapping segments to maintain context.

    Embedding: Text chunks are converted into numerical vectors using Google's embedding models.

    Retrieval: When a user asks a question, the system finds the most relevant paragraphs from the textbook.

    Generation: The Gemini SLM reads the question + the relevant paragraphs to provide a grounded, accurate answer.

⚙️ Installation & Setup

If you want to run this project locally:

    Clone the repository:
    Bash

    git clone https://github.com/YOUR_USERNAME/ScienceX-search-engine.git
    cd ScienceX-search-engine

    Install dependencies:
    Bash

    pip install -r requirements.txt

    Set up your API Key:
    Create a .streamlit/secrets.toml file and add your key:
    Ini, TOML

    GOOGLE_API_KEY = "google_api_key_here"

    Run the app:
    Bash

    streamlit run app.py

📂 File Structure

    app.py: The main application logic and UI.

    requirements.txt: List of necessary Python libraries.

    jesc101.pdf - jesc113.pdf: The source textbook chapters.

    faiss_index/: Local folder created after first run to store searchable data.

🤝 Contributing

This is an educational project. Feel free to fork this repository and add features like multi-language support or support for other subjects!

I am also going to add sources such as NCERT answer keys.
<img width="1919" height="914" alt="image" src="https://github.com/user-attachments/assets/e4e291c8-7c97-48d4-a681-78b67deeaf69" />
