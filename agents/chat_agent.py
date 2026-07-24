import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.groq_client import ask_groq
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def run_chat_agent(
    question: str,
    chat_history: list,
    profile: dict,
    df_sample: str,
    insights: str = "",
    chart_specs: list = [],
    profile_analysis: str = "",
    cleaning_report: list = [],
    anomalies: list = [],
    use_rag: bool = False,        # NEW: whether to retrieve from knowledge documents
) -> str:

    # ── RAG: Retrieve relevant knowledge if documents exist ──────────────────
    rag_context = ""
    if use_rag:
        try:
            from rag.pipeline import query_rag_pipeline, get_pipeline_status
            status = get_pipeline_status()
            if status["ready"]:
                chunks = query_rag_pipeline(question, top_k=3)
                if chunks:
                    parts = []
                    for i, chunk in enumerate(chunks):
                        parts.append(
                            f"[Knowledge Source {i + 1}: {chunk['filename']}]\n{chunk['text']}"
                        )
                    rag_context = "\n\n---\n\n".join(parts)
                    print(f"[ChatAgent] RAG retrieved {len(chunks)} chunk(s) for context.")
            else:
                print("[ChatAgent] RAG pipeline not ready — skipping retrieval.")
        except Exception as e:
            print(f"[ChatAgent] RAG retrieval failed — skipping. Error: {e}")

    # ── Build system prompt ───────────────────────────────────────────────────
    system_prompt = f"""You are an expert data analyst assistant analyzing a dataset.

Dataset context:
- Rows: {profile['rows']}, Columns: {profile['columns']}
- Column names: {list(profile['column_details'].keys())}
- Target column: {profile['guessed_target']}
- Numeric columns: {profile['numeric_cols']}
- Categorical columns: {profile['categorical_cols']}

Sample data:
{df_sample}

Profile Agent Analysis:
{profile_analysis}

Cleaning Agent Report:
{cleaning_report}

Anomalies Detected:
{anomalies}

Visualization Agent - Charts Generated:
{chart_specs}

Insight Agent - Business Insights:
{insights}"""

    # ── Append RAG context if available ──────────────────────────────────────
    if rag_context:
        system_prompt += f"""

Retrieved Knowledge Documents:
The following context was retrieved from uploaded knowledge documents.
Use it to enrich your answer when relevant to the question.

{rag_context}"""

    system_prompt += """

Rules:
- You have full context from all agents above
- Never say you cannot see charts — you know exactly which charts were generated
- Never say you cannot analyze data — you have the sample and all agent outputs
- Answer questions by combining findings from all agents when relevant
- If knowledge documents were provided, use them to add policy or domain context
- Reference specific column names, numbers, and agent findings
- Be concise and business-oriented
- If asked a followup, use conversation history for context"""

    # ── Build messages with full history ──────────────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content