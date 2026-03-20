import argparse
import json
import os
import sys
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

INDEX_PATH   = "./fo_faiss.index"
METADATA_PATH= "./fo_metadata.json"
EMBED_MODEL  = "all-MiniLM-L6-v2"
GROQ_MODEL   = "llama-3.3-70b-versatile"
TOP_K        = 8
MAX_TOKENS   = 1500

SYSTEM_PROMPT = """You are a family office intelligence analyst. Answer the user's question using ONLY the retrieved records below.

Rules:
- Only use facts from the retrieved records. Never invent data.
- If records do not contain enough information, say so clearly.
- Be specific — name actual family offices, decision makers, AUM figures, check sizes.
- Keep answers concise and actionable — this is intelligence for a deal team.
- If asked about contact details, include them if present in the records."""


def load_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("\n  ERROR: Vector store not found. Run step1_ingest.py first.\n")
        sys.exit(1)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)
    return index, store


def retrieve(query, index, store, model, top_k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)

    retrieved = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        retrieved.append({
            "id":       store["ids"][idx],
            "document": store["documents"][idx],
            "metadata": store["metadatas"][idx],
            "score":    round(float(score), 4),
        })
    return retrieved


def generate_answer(query, retrieved, groq_client):
    context = "\n\n".join([
        f"--- Record {i+1} (ID: {r['id']}, Relevance: {r['score']}) ---\n{r['document']}"
        for i, r in enumerate(retrieved)
    ])
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Records:\n\n{context}\n\n---\n\nQuestion: {query}\n\nAnswer based only on the records above."}
        ]
    )
    return response.choices[0].message.content


def display_results(query, retrieved, answer):
    print()
    print("=" * 70)
    print(f"  QUERY: {query}")
    print("=" * 70)
    print(f"\n  Retrieved {len(retrieved)} records:")
    for r in retrieved:
        m = r["metadata"]
        print(f"  [{r['score']:.3f}] {m.get('fo_name','?')[:45]} | {m.get('region','?')} | AUM: ${m.get('aum_millions',0):,.0f}M")
    print()
    print("-" * 70)
    print("  ANSWER")
    print("-" * 70)
    print()
    for line in answer.split("\n"):
        print(f"  {line}")
    print()
    print("=" * 70)


def interactive_mode(index, store, model, groq_client, top_k):
    print()
    print("=" * 70)
    print("  FO INTELLIGENCE — NATURAL LANGUAGE QUERY INTERFACE")
    print(f"  Model: {GROQ_MODEL} (Groq — free)")
    print("  Type your question. Type 'exit' to quit.")
    print("=" * 70)

    while True:
        print()
        query = input("  Your question: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("  Goodbye.")
            break
        print("  Retrieving...")
        retrieved = retrieve(query, index, store, model, top_k)
        print("  Generating answer...")
        answer = generate_answer(query, retrieved, groq_client)
        display_results(query, retrieved, answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",  default=None)
    parser.add_argument("--top_k",  type=int, default=TOP_K)
    args = parser.parse_args()

    print("\n  Loading vector store and embedding model...")
    index, store = load_store()
    model = SentenceTransformer(EMBED_MODEL)
    print(f"  {index.ntotal} records loaded")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n  ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)

    if args.query:
        retrieved = retrieve(args.query, index, store, model, args.top_k)
        answer    = generate_answer(args.query, retrieved, groq_client)
        display_results(args.query, retrieved, answer)
    else:
        interactive_mode(index, store, model, groq_client, args.top_k)


if __name__ == "__main__":
    main()
