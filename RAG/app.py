#pip install -q sentence-transformers faiss-cpu transformers accelerate datasets

#importing libraries
import os
import math
from typing import List 

import torch
from sentence_transformers import SentenceTransformer
import faiss

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ", device)

# Data setup
#Datasets
import pandas as pd

data = {
    'Shipment_ID': ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008'],
    'Origin': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'Seattle', 'Denver', 'Atlanta'],
    'Destination': ['San Francisco', 'Dallas', 'Boston', 'Phoenix', 'Orlando', 'Portland', 'Austin', 'Charlotte'],
    'Weight_kg': [1500, 800, 2500, 1200, 300, 1800, 600, 2000],
    'Volume_m3': [10.5, 5.2, 18.0, 7.8, 2.1, 12.3, 3.5, 15.0],
    'Status': ['In Transit', 'Delivered', 'Pending', 'Delayed', 'In Transit', 'Delivered', 'Pending', 'In Transit'],
    'Delivery_Date': ['2023-10-25', '2023-10-20', '2023-11-01', '2023-10-28', '2023-10-26', '2023-10-22', '2023-11-03', '2023-10-29'],
    'Vehicle_Type': ['Truck', 'Van', 'Truck', 'Van', 'Truck', 'Truck', 'Van', 'Truck'],
    'Vehicle_Capacity_kg': [2000, 1000, 3000, 1500, 1000, 2500, 800, 2500],
    'Current_Location': ['Ohio', 'Texas', 'Indiana', 'Arizona', 'Georgia', 'Oregon', 'Kansas', 'Tennessee'],
    'Route_Distance_km': [4500, 2000, 1500, 1800, 1000, 300, 2000, 500],
    'Route_Time_hours': [60, 28, 20, 24, 15, 5, 30, 8],
    'Route_Stops': [3, 1, 2, 2, 1, 0, 2, 1],
    'Warehouse_Name': ['Main Hub NY', 'LA Distribution', 'Chicago Depot', 'Houston Hub', 'Miami Logistics', 'Seattle Warehouse', 'Denver Stock', 'Atlanta Central'],
    'Item_SKU': ['A101', 'B202', 'C303', 'D404', 'E505', 'F606', 'G707', 'H808'],
    'Stock_Level': [100, 250, 75, 120, 300, 50, 180, 90],
    'Warehouse_Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami', 'Seattle', 'Denver', 'Atlanta']
}

print("Logistics data dictionary populated.")
df = pd.DataFrame(data)
df.to_csv("Logistics_docs.csv", index=False)
df.head()

from typing import List
import pandas as pd

#chunking and preprocessing

def chunk_text(text: str, chunk_size: int=500) -> List[str]:
  if not isinstance(text,str):
    return[]
  max_chars = chunk_size
  text = text.strip()
  chunks = []
  start = 0
  while start < len(text):
    chunks.append(text[start:start+max_chars].strip())
    start += max_chars
  return chunks

df = pd.read_csv("Logistics_docs.csv")
df.head()
passages = []
for idx, row in df.iterrows():
  # Construct a comprehensive text for each document (shipment)
  full_text = " ".join([f"{col}: {val}" for col, val in row.items() if col not in ['Shipment_ID']])
  
  # Use Shipment_ID as doc_id and Origin-Destination as title
  doc_id = row['Shipment_ID']
  title = f"Shipment from {row['Origin']} to {row['Destination']}"

  chunks = chunk_text(full_text, chunk_size=500)
  if not chunks:
    chunks = [" "] # Ensure there's at least one chunk even if text is empty

  for i,c in enumerate(chunks):
    passages.append({
        "doc_id": doc_id,
        "title": title,
        "chunk_id": f"{doc_id}_chunk_{i}",
        "text": c
        })
print("Created passages: ", len(passages))
passages[:3]

#building faiss and embeddings
import numpy as np
EMBED_MODEL = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBED_MODEL, device=device if device=="cuda" else "cpu")


texts = [p["text"] if p["text"] else "" for p in passages]
embeddings = embed_model.encode(texts, convert_to_tensor=True, show_progress_bar=True, convert_to_numpy=True)

#normalize for cosine similarity using inner product
def normalize(v: np.ndarray):
  norms = np.linalg.norm(v, axis=1, keepdims=True)
  norms[norms==0] = 1.0
  return v / norms

embeddings = normalize(embeddings)
d=embeddings.shape[1]

index = faiss.IndexFlatIP(d)
index.add(embeddings)
print("FAISS index size:", index.ntotal)
meta = passages

#retrieval fucntion
def retrieve(query: str, top_k: int = 4):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = normalize(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        m = meta[idx]
        results.append({
            "score": float(score),
            "doc_id": m["doc_id"],
            "title": m["title"],
            "chunk_id": m["chunk_id"],
            "text": m["text"]
        })
    return results

# quick test
print(retrieve("What documents are needed for customs clearance?"))

#loading a HF generator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

print("Loading generation model:", model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#Prompt builder and HF generation function

def build_Promt(query: str, results: List[dict]) -> str:
  ctx_text = "\n\n".join([f"{c['title']} ({c['doc_id']}): {c['text']}" for c in results])
  prompt = (
      "You are a helpful logistics assistant. Use ONLY the provided context to answer the user's question.\n\n"
      f"Context:\n{ctx_text}\n\n"
      f"User question: {query}\n\n"
      "Answer concisely and, if possible, mention the document ids that support your answer."
  )
  return prompt

def rag_answer_hf(query: str, top_k: int = 4, max_new_tokens: int = 150):
  contexts = retrieve(query, top_k)
  if not contexts:
    return "no relavant documents found"

  prompt = build_Promt(query, contexts)
  inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
  if device == "cuda":
    inputs = {k: v.to(device) for k,v in inputs.items()}

  outputs =  gen_model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      do_sample=False,  # deterministic; set True for creative answers
      temperature=0.0,
      num_return_sequences=1
  )
  decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return {"answer": decoded.strip(), "contexts": contexts}

# quick test of RAG function
q = "Explain required customs documents."
res = rag_answer_hf(q, top_k=3)
print("QUESTION:\n", q)
print("\nANSWER:\n", res["answer"])
print("\nSOURCES (retrieved):")
for c in res["contexts"]:
    print(f"- {c['doc_id']}: {c['title']} -> {c['text']}")
