from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from algorithms import NgramAutocomplete, CustomBM25

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print("Loading Embedding Model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading LLM (Qwen 1.5B)...")
llm_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.float32, 
    device_map="auto"
)

stored_notes = []
embeddings_db = None
ngram_model = NgramAutocomplete()
bm25_model = CustomBM25()

class NoteInput(BaseModel):
    text: str

class QueryInput(BaseModel):
    query: str

def chunk_text(text, chunk_size=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

@app.post("/add_note")
async def add_note(note: NoteInput):
    global stored_notes, embeddings_db
    chunks = chunk_text(note.text)
    stored_notes.extend(chunks)
    
    new_embeddings = embedder.encode(chunks)
    if embeddings_db is None:
        embeddings_db = new_embeddings
    else:
        embeddings_db = np.vstack((embeddings_db, new_embeddings))
        
    ngram_model.train(chunks)
    bm25_model.fit(stored_notes)
    
    return {"message": "Note added and indexed successfully!"}

@app.get("/autocomplete")
async def autocomplete(q: str):
    suggestion = ngram_model.suggest_next_word(q)
    if not suggestion:
        return {"suggestion": ""}
    return {"suggestion": f"{q} {suggestion}".strip()}

@app.post("/chat")
async def chat(query: QueryInput):
    if not stored_notes:
        raise HTTPException(status_code=400, detail="No notes added yet.")
        
    q_str = query.query
    
    q_embed = embedder.encode([q_str])
    similarities = np.dot(embeddings_db, q_embed.T).flatten()
    dense_top_idx = np.argsort(similarities)[-2:][::-1]
    
    bm25_results = bm25_model.search(q_str, top_k=2)
    
    context_chunks = set([stored_notes[i] for i in dense_top_idx] + [res[1] for res in bm25_results])
    context_text = "\n- ".join(context_chunks)
    
    prompt = f"Use the following notes to answer the question.\nNotes:\n- {context_text}\n\nQuestion: {q_str}\nAnswer:"
    
    result = llm_pipeline(prompt, max_new_tokens=150, truncation=True)
    generated_text = result[0]['generated_text'][len(prompt):]
    
    return {
        "answer": generated_text.strip(),
        "context_used": list(context_chunks)
    }
