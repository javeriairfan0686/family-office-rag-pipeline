import json
import os
import sys
import datetime
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

INDEX_PATH    = "./fo_faiss.index"
METADATA_PATH = "./fo_metadata.json"
EMBED_MODEL   = "all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.3-70b-versatile"
TOP_K         = 8
MAX_TOKENS    = 1500
OUTPUT_FILE   = "FO_RAG_Demo_Results.txt"

DEMO_QUERIES = [
    {"id": "Q1", "query": "Which family offices focus on AI or technology with check sizes above $10M?",               "why": "Sector + financial filter"},
    {"id": "Q2", "query": "Who are the most active SFO decision makers in Europe and what do they invest in?",        "why": "Region + FO type + executive intelligence"},
    {"id": "Q3", "query": "Which family offices have high co-investment appetite and are open to ESG deals?",          "why": "Multi-field filter retrieval"},
    {"id": "Q4", "query": "Show me family offices in the Middle East with AUM above $10 billion",                     "why": "Geographic + AUM threshold"},
    {"id": "Q5", "query": "Which second-generation G2 family offices in Asia focus on real estate or infrastructure?","why": "Succession stage + region + sector"},
]

SYSTEM_PROMPT = """You are a family office intelligence analyst. Answer using ONLY the retrieved records. Be specific, name actual family offices and decision makers. Never invent data. Keep answers concise and actionable."""


def retrieve(query, index, store, model, top_k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    return [{"id": store["ids"][idx], "document": store["documents"][idx],
             "metadata": store["metadatas"][idx], "score": round(float(score), 4)}
            for score, idx in zip(scores[0], indices[0]) if idx != -1]


def generate_answer(query, retrieved, groq_client):
    context = "\n\n".join([f"--- Record {i+1} (ID: {r['id']}, Relevance: {r['score']}) ---\n{r['document']}" for i, r in enumerate(retrieved)])
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL, max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Records:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer based only on records above."}
        ]
    )
    return response.choices[0].message.content


def run_demo():
    print()
    print("=" * 70)
    print("  FO RAG PIPELINE — DEMO")
    print("  5 example queries | FAISS + Groq Llama 3.3 (free)")
    print("=" * 70)

    if not os.path.exists(INDEX_PATH):
        print("\n  ERROR: Run step1_ingest.py first.\n")
        sys.exit(1)

    print("\n  Loading vector store and model...")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  {index.ntotal} records ready")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n  ERROR: GROQ_API_KEY not set.\n")
        sys.exit(1)
    groq_client = Groq(api_key=api_key)

    lines = []
    lines.append("=" * 70)
    lines.append("  INTERNATIONAL FAMILY OFFICE INTELLIGENCE")
    lines.append("  RAG Pipeline — Demo Query Results")
    lines.append(f"  Generated: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}")
    lines.append(f"  Dataset: 235 records  |  LLM: {GROQ_MODEL} via Groq (free)")
    lines.append(f"  Embeddings: {EMBED_MODEL}  |  Vector DB: FAISS  |  Top-K: {TOP_K}")
    lines.append("=" * 70)

    for demo in DEMO_QUERIES:
        print(f"\n  Running {demo['id']}: {demo['query'][:55]}...")
        retrieved = retrieve(demo["query"], index, store, model)
        answer    = generate_answer(demo["query"], retrieved, groq_client)
        print(f"  Done — {len(retrieved)} records retrieved")

        lines.append("")
        lines.append("-" * 70)
        lines.append(f"  {demo['id']}  {demo['query']}")
        lines.append(f"  Purpose: {demo['why']}")
        lines.append("-" * 70)
        lines.append(f"\n  RETRIEVED RECORDS ({len(retrieved)}):")
        for r in retrieved:
            m = r["metadata"]
            lines.append(f"  [{r['score']:.3f}] {m.get('fo_name','?')} | {m.get('region','?')} | AUM: ${m.get('aum_millions',0):,.0f}M | Confidence: {m.get('confidence','?')}")
        lines.append("\n  ANSWER:\n")
        for line in answer.split("\n"):
            lines.append(f"  {line}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF DEMO")
    lines.append("=" * 70)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Done! Results saved to: {OUTPUT_FILE}")
    print()


if __name__ == "__main__":
    run_demo()
