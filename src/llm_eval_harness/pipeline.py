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
        # Greedy decoding with a fixed seed. Sampling made the same question
        # answer differently every run, which moved the refusal metric by 0.25
        # across five identical runs and left it unable to tell a regression
        # from noise. Nothing here wants creative variation anyway: the answer
        # is supposed to be whatever the retrieved context supports.
        options={"temperature": 0, "seed": 0},
    )
    return {
        "question": question,
        "answer": response.message.content,
        "contexts": contexts,
    }
