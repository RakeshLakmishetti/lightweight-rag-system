from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# config
docs_dir_path = r"docs_dir_path"
vector_db_path = r"vector_db_path"
collection_name = "document_collection"

# ---- 1. Load PDFs ----
loader = DirectoryLoader(
    path=docs_dir_path,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# ---- 2. Split text ----
splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
chunks = splitter.split_documents(documents)

# ---- 3. Create TF-IDF embeddings ----
vectorizer = TfidfVectorizer()
texts = [chunk.page_content for chunk in chunks]

# Fit + transform (embedding)
vectors = vectorizer.fit_transform(texts).toarray().astype(np.float32)

# ---- 4. Store in Chroma ----
vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=vector_db_path
)

# Add embeddings manually
vector_store._collection.add(
    embeddings=vectors,
    documents=texts,
    metadatas=[chunk.metadata for chunk in chunks],
    ids=[str(i) for i in range(len(chunks))]
)

print("Vector DB created successfully with TF-IDF embeddings!")

# ---------------------------------------
# SAVE TF-IDF VECTORIZER (IMPORTANT)
# ---------------------------------------
import pickle
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


print("TF-IDF vectorizer saved!")
