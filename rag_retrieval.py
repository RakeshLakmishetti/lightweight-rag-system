from groq import Groq
import numpy as np
import pickle
from langchain_chroma import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer

# load vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# load vector DB
vector_store = Chroma(
    collection_name="document_collection",
    persist_directory=r"C:\Users\RakeshLakmishetti\Desktop\New folder (2)\vector_db"
)

# question
query = "What is the First name?"

# embed query
query_vec = vectorizer.transform([query]).toarray().astype(np.float32)

# retrieve top chunk
results = vector_store._collection.query(
    query_embeddings=query_vec,
    n_results=2
)

# combine retrieved text
context = "\n".join(results["documents"][0])

# ask Groq to answer concisely
client = Groq(api_key="your_api_key")
prompt = f"Answer only about the given question and don't tell anything extra or anything less, using the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"

reply = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}]
)

print(reply.choices[0].message.content)
