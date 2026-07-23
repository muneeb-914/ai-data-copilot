from rag.pipeline import build_rag_pipeline, query_rag_pipeline, get_pipeline_status

# Step 1: Build
print("=== BUILDING PIPELINE ===")
result = build_rag_pipeline()
print(result)

# Step 2: Status check
print("\n=== PIPELINE STATUS ===")
status = get_pipeline_status()
print(status)

# Step 3: Query 1 - should return churn related chunks
print("\n=== QUERY 1: Churn ===")
chunks = query_rag_pipeline("what is the churn risk rule?", top_k=2)
for c in chunks:
    print(f"File: {c['filename']} | Score: {c['score']:.4f}")
    print(f"Text: {c['text'][:120]}")
    print()

# Step 4: Query 2 - should return cleaning policy chunks
print("\n=== QUERY 2: Data Cleaning ===")
chunks = query_rag_pipeline("how should missing values be handled?", top_k=2)
for c in chunks:
    print(f"File: {c['filename']} | Score: {c['score']:.4f}")
    print(f"Text: {c['text'][:120]}")
    print()

# Step 5: Query 3 - should return customer segmentation chunks
print("\n=== QUERY 3: Customer Segments ===")
chunks = query_rag_pipeline("what are the gold customer rules?", top_k=2)
for c in chunks:
    print(f"File: {c['filename']} | Score: {c['score']:.4f}")
    print(f"Text: {c['text'][:120]}")
    print()


# Stage 4: Full RAG answer test
from rag.pipeline import answer_with_rag

print("\n=== QUERY + GROQ ANSWER 1 ===")
result = answer_with_rag("what should happen when a customer has not purchased in 90 days?")
print("Answer:", result["answer"])
print("Sources used:")
for s in result["sources"]:
    print(f"  - {s['filename']} | chunk {s['chunk_index']} | score {s['score']:.4f}")

print("\n=== QUERY + GROQ ANSWER 2 ===")
result = answer_with_rag("how should missing values in numerical columns be handled?")
print("Answer:", result["answer"])
print("Sources used:")
for s in result["sources"]:
    print(f"  - {s['filename']} | chunk {s['chunk_index']} | score {s['score']:.4f}")