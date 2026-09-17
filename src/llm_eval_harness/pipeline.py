import ollama

from llm_eval_harness.store import hybrid_search

MODEL = "gemma4:latest"
K = 5

# Dense retrieval alone served no part of the gold span in the top five for ten
# of the twenty-six answerable questions. The generator then did the only two
# things it could: it declined on six of them, correctly, and answered four
# from chunks that did not support the answer. Fusing BM25 into the ranking
# fixes four of the ten and costs two (scripts/compare_retrievers.py).
#
# The misses were not spread evenly. They collected on questions that name a
# document - "the scope of NIST SP 800-78-5" - where an embedding blurs the
# identifier into every neighbouring document number and exact word matching
# does not.
RETRIEVE = hybrid_search
SYSTEM = "Give answers only based on context, if answer is not in context, say that answer is missing."


def answer(question):
    contexts = RETRIEVE(question, k=K)
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
