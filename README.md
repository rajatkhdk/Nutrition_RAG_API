# Food Nutrition RAG API

A Retrieval-Augmented Generation (RAG) based nutrition assistant that answers food nutrition queries using an Indian food dataset and a large language model.

Built with:
- FastAPI backend
- Chroma vector database
- Sentence Transformers embeddings
- Groq LLM (LLaMA 3)
- CSV nutrition dataset

---

## Features

- Semantic search over food dataset  
- Retrieval-Augmented Generation (RAG)  
- Accurate nutrition extraction from context  
- JSON structured responses  
- FastAPI REST API  
- Persistent vector database (Chroma)  
- Ready for Web / Flutter / Mobile apps  

---

## 🧠 How It Works (Architecture)

User Query  
- Embed query  
- Search similar foods in vector DB  
- Retrieve top-k matches  
- Send as context to LLM  
- LLM extracts nutrition  
- Return structured JSON  

CSV -> Embeddings -> ChromaDB -> Retrieval -> LLM -> API Response


---

## ⚙️ Installation

### 1. Clone repo

```bash
git clone https://github.com/rajatkhdk/Nutrition_RAG_API
cd nutrition-rag-api
```

### 2. Install dependencies
```bash
uv add -r requirement.txt
```

### 3. Add API key

Create .env
```bash
GROQ_API_KEY=your_api_key
```

### 4. Run Server
``` bash
uvicorn app:app --reload
```
Server starts at: http://127.0.0.1:8000

### 5. Testing:
#### i. Swagger UI:
- Open: http://127.0.0.1:8000/docs
- Click: /get_nutrition
- Click: Try it out

### Example Request:
``` json
{
  "query": "Cream of tomato soup"
}
```

### Example Response:
``` json
{
  "Name": "Cream of Tomato Soup",
  "Serving Size": "1 serving",
  "Serving unit": "100g",
  "Calories (kcal)": 98,
  "Protein (g)": 4.6,
  "Carbs (g)": 3.9,
  "Fats (g)": 13.1,
  "Fiber (g)": 1.4
}
```

## Core Components
rag_pipeline.py
- CSV -> Documents
- Sentence embeddings
- Vector storage with Chroma
- Similarity retrieval

app.py
- FastAPI server
- LLM prompt engineering
- JSON structured responses