# Hospital Saathi — AI-Powered Hospital Assistant

Hospital Saathi is an AI-powered hospital information assistant that helps users find doctors, explore hospital facilities, locate rooms and departments, and retrieve doctor-profile information through a conversational interface.

## Features

- **Doctor search:** Find doctors by specialty and expertise.
- **Hospital information:** Retrieve information about services, departments, facilities, and policies.
- **Hospital navigation:** Find locations such as doctor cabins, the ICU, and the canteen.
- **Doctor profiles:** Retrieve qualifications, experience, and synthetic patient-experience examples.
- **RAG-based responses:** Uses relevant document chunks to ground responses in the supplied hospital data.
- **Conversation persistence:** SQLite-backed LangGraph checkpointing supports continuing previous chat threads.
- **Chat interface:** Streamlit provides the web-based conversational UI.

> **Demo-data notice:** The hospital, doctor, and patient-experience content used in this prototype is synthetic. Patient feedback is fictional and must not be treated as real or verified reviews. This project is a demonstration information assistant, not a diagnostic or treatment system.

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit, custom CSS | Web-based chat interface |
| Frontend development | AI-assisted development | AI tools were used to assist with generating and refining the Streamlit UI and styling |
| LLM | NVIDIA-hosted chat model via `ChatNVIDIA` | Tool selection and natural-language response generation |
| Orchestration | LangGraph | Stateful workflow, conditional routing, and tool execution |
| AI integrations | LangChain | Tools, document loading, embeddings, and retrieval integration |
| Embedding model | Hugging Face `BAAI/bge-m3` | Converts document chunks and queries into vector embeddings |
| Vector search | FAISS | Semantic search over hospital document collections |
| PDF ingestion | `PyPDFLoader` | Loads hospital PDF documents |
| Text splitting | `RecursiveCharacterTextSplitter` | Splits documents into overlapping chunks |
| Conversation persistence | SQLite + LangGraph `SqliteSaver` | Stores graph checkpoints and conversation state |
| Configuration | `python-dotenv` | Loads environment variables such as API keys |

## RAG Workflow

1. Hospital PDFs are loaded with `PyPDFLoader`.
2. Documents are split into chunks using a chunk size of 1,000 characters and overlap of 200 characters.
3. `BAAI/bge-m3` generates embeddings for the chunks.
4. Four FAISS indexes are created for the separate document collections.
5. LangGraph routes each user query to the appropriate retrieval tool.
6. The retriever returns the top three relevant chunks.
7. The NVIDIA-hosted LLM uses the conversation and retrieved information to generate a response.
8. SQLite checkpointing preserves conversation state for each thread.

### Document collections

- `hospital_info.pdf` — hospital services, facilities, departments, and policies.
- `doctor_search.pdf` — doctor names, qualifications, specialties, and expertise.
- `hospital_route.pdf` — buildings, floors, rooms, and facility directions.
- `doctor_profile.pdf` — doctor profiles and fictional patient-experience examples.

## Project Structure

```text
hospital_chatbot/
├── app.py
├── backend.py
├── vectorbase.py
├── retriever.py
├── tools.py                 # If tools are maintained in a separate module
├── data2/
│   ├── hospital_info.pdf
│   ├── doctor_search.pdf
│   ├── hospital_route.pdf
│   └── doctor_profile.pdf
├── faiss_db/                # Generated local FAISS indexes
├── chat_history.db           # SQLite conversation checkpoints
├── .env                      # Local API key; do not commit
├── .gitignore
└── requirements.txt
```

The exact files may vary depending on how the project modules are organized. In the current backend example, the four tools are defined directly in `backend.py`; if so, `tools.py` is optional.

## Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd hospital_chatbot
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

If `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Otherwise, install the main packages:

```bash
pip install streamlit langchain langchain-community langchain-core langchain-text-splitters langchain-huggingface langchain-nvidia-ai-endpoints langgraph langgraph-checkpoint-sqlite faiss-cpu sentence-transformers pypdf python-dotenv
```

### 4. Configure the NVIDIA API key

Create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_nvidia_api_key
```

Use a valid NVIDIA-supported chat model name in the `ChatNVIDIA` configuration in `backend.py`. Never commit the `.env` file or expose API keys publicly.

### 5. Add the hospital documents

Place the four PDF files in the `data2/` directory, using the filenames shown above.

### 6. Build or load the FAISS indexes

Run the vector-store script:

```bash
python vectorbase.py
```

On the first run, the embedding model is downloaded if needed and the indexes are created. If your implementation saves indexes locally, subsequent runs can load them. If a source PDF changes, rebuild its corresponding index.

### 7. Launch the Streamlit application

```bash
streamlit run app.py
```

## Storage Notes

- **FAISS** stores the local vector indexes used for semantic document search.
- **SQLite** stores LangGraph checkpoints and conversation state.
- Neither is currently an appointment-booking database.
- Generated indexes and local chat history should generally remain out of version control.

## Current Limitations

- Uses synthetic hospital and doctor data.
- Does not provide real-time doctor availability or appointment booking.
- Does not integrate with a live hospital management system.
- Requires a valid NVIDIA API key and internet access for hosted LLM calls.
- Responses depend on the quality and coverage of the supplied documents.

## Future Scope

- Appointment booking with live availability and double-booking prevention.
- PostgreSQL or another relational database for operational data.
- Secure integration with hospital management systems.
- Multilingual support for English, Hindi, Punjabi, and other languages.
- Voice input and text-to-speech accessibility features.
- Hybrid retrieval, reranking, and systematic retrieval evaluation.
- Authentication, role-based access, audit logs, backups, monitoring, and secure cloud deployment.

## Responsible Use

Hospital Saathi is a prototype for retrieving hospital information. It is not a substitute for a qualified healthcare professional and must not be used to diagnose conditions or prescribe treatment. Potential emergencies require immediate medical assistance. Real-world deployment would require approved data, privacy and security controls, clinical review, and appropriate hospital authorization.

## Acknowledgment

AI-assisted development tools were used to help design, generate, and refine parts of the Streamlit frontend and its CSS styling. The project was developed as an individual project, and the implementation and testing are maintained by the author.

## License

Add a license before distributing or reusing this project publicly.
