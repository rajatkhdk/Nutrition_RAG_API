
import os
from langchain_core.documents import Document
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Any

# --- CSV Processing ---
def process_csv(csv_path: str) -> List[Document]:
    df = pd.read_csv(csv_path)
    documents = []
    for i, row in df.iterrows():
        content = (
            f"Food: {row['Dish Name']}\n"
            f"Calories (kcal): {row['Calories (kcal)']}\n"
            f"Carbohydrates (g): {row['Carbohydrates (g)']}\n"
            f"Protein (g): {row['Protein (g)']}\n"
            f"Fats (g): {row['Fats (g)']}\n"
            f"Fiber (g): {row['Fibre (g)']}"
        )
        metadata = row.to_dict()
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

# --- Embeddings ---
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def generate_embeddings(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False)

# --- Vector Store ---
class VectorStore:
    def __init__(self, collection_name="food_calorie", persist_dir="./data/vector_store"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "food calorie document embeddings for RAG"}
        )

    def add_documents(self, documents: List[Any], embeddings):
        ids, metadatas, texts, embed_list = [], [], [], []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadatas.append({**doc.metadata, "doc_index": i})
            texts.append(doc.page_content)
            embed_list.append(emb.tolist())
        self.collection.add(ids=ids, embeddings=embed_list, metadatas=metadatas, documents=texts)

# --- RAG Retrieval ---
class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5):
        query_emb = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for doc, meta, dist, doc_id in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
                results['ids'][0]
            ):
                similarity = 1 - dist
                retrieved_docs.append({
                    "id": doc_id,
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity
                })
        return retrieved_docs

# --- Initialize ---
documents = process_csv("./data/csv/indian_food.csv")
embedding_manager = EmbeddingManager()
texts = [doc.page_content for doc in documents]
embeddings = embedding_manager.generate_embeddings(texts)
vector_store = VectorStore()
vector_store.add_documents(documents, embeddings)
rag_retriever = RAGRetriever(vector_store, embedding_manager)
