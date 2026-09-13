import ollama

from llm_eval_harness.store import search

MODEL = "gemma4:latest"
K = 5
SYSTEM = "Give answers only based on context, if answer is not in context, say that answer is missing."


def answer(question):
    contexts = search(question, k=K)
    blocks = [f"[{c['source']}]\n{c['text']}" for c in contexts]
    context_text = "\n\n---\n\n".join(blocks)
    prompt = f"""{context_text}

Question: {question}"""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return {
        "question": question,
        "answer": response.message.content,
        "contexts": contexts,
    }
