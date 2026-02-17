from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import rag_retriever  # import RAG retriever from your module
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import json

# --- Load environment variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Initialize LLM ---
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1,
    max_tokens=1024
)

# --- FastAPI ---
app = FastAPI(title="Nutrition RAG API")

# --- CORS for Flutter / Web ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your app domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Schema ---
class QueryRequest(BaseModel):
    query: str

# --- RAG Endpoint ---
@app.post("/get_nutrition")
def get_nutrition(request: QueryRequest):
    results = rag_retriever.retrieve(request.query, top_k=3)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""
    if not context:
        return {"result": '{"error": "No nutritional data found"}'}

    prompt = f"""
You are a nutrition extraction assistant. Use ONLY the text under "Context" when it contains direct nutrition information for the requested food. If the context does not contain explicit nutrition values, you MAY estimate values using common nutrition databases (USDA / typical food-composition averages). Follow these rules precisely:

1. Use the Context first. Only infer/estimate when the context lacks the item or numeric values.
2. Provide a short, normalized Name for the food (use the Query if the context has no explicit name).
3. Serving Size must be a human-readable string (e.g. "1 paratha (100 g)", "100 g", "1 cup (240 ml)").
4. Numeric fields:
   - "Calories (kcal)" → integer (round to nearest whole number).
   - "Protein (g)", "Carbs (g)", "Fats (g)", "Fiber (g)" → numeric with one decimal place (round to one decimal).
5. If you must estimate, pick values that reflect a typical serving and base them on common food composition knowledge.
6. If a numeric value is completely unavailable and you cannot reasonably estimate, set that numeric field to 0.
7. **Output ONLY** the JSON object below — no explanations, no surrounding text, no markdown, no extra fields, and no trailing commas.
Context:
{context}

Question: {request.query}

Return the answer in JSON format exactly as below (no extra text):

{{
    "Name": "",
    "Serving Size": "",
    "Serving unit": "",
    "Calories (kcal)": 0,
    "Protein (g)": 0,
    "Carbs (g)": 0,
    "Fats (g)": 0,
    "Fiber (g)": 0
}}

Answer:
"""
    response = llm.invoke(prompt)
    
    try:
        parsed = json.loads(response.content)   # ⭐ convert to JSON
        return parsed
    except:
        return {"error": "Invalid LLM output", "raw": response.content}