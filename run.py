from app.services.ingestion import load_file, chunk_documents
from pathlib import Path
from app.rag.vectorstore import add_documents

doc = load_file(Path("data/sample_kb/company_handbook.md"))
docs = chunk_documents(doc)

add_documents(docs)