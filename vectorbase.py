
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent
DATA_DIR=BASE_DIR/"data2"
DB_DIR=BASE_DIR/"faiss_db"

model=HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

def create_vectorstore(filename,folder):
    pdf_path=DATA_DIR/filename
    db_path=DB_DIR/folder
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if (db_path/"index.faiss").exists() and (db_path/"index.pkl").exists():
        print(f"Loading saved index: {folder}")
        return FAISS.load_local(str(db_path),model,allow_dangerous_deserialization=True)
    print(f"Creating index: {folder}")
    docs=PyPDFLoader(str(pdf_path)).load()
    chunks=splitter.split_documents(docs)
    if not chunks:
        raise ValueError(f"No text extracted from {filename}")
    db=FAISS.from_documents(chunks,model)
    db_path.mkdir(parents=True,exist_ok=True)
    db.save_local(str(db_path))
    print(f"Saved index: {db_path}")
    return db

hospital_vectorstore=create_vectorstore("hospital_info.pdf","hospital")
doctor_vectorstore=create_vectorstore("doctor_search.pdf","doctor")
route_vectorstore=create_vectorstore("hospital_route.pdf","route")
profile_vectorstore=create_vectorstore("doctor_profile.pdf","profile")

print("All four FAISS vector stores are ready.")
