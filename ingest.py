import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load all docs from data/
docs = []
for file in os.listdir("data/"):
    filepath = os.path.join("data/", file)
    if file.endswith(".txt"):
        loader = TextLoader(filepath)
    elif file.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    else:
        continue
    docs.extend(loader.load())
    print(f"Loaded: {file}")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# Embed & store (local Chroma DB)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_db"
)
print("Indexing complete! KB ready.")