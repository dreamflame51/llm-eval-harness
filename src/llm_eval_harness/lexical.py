"""
Word matching next to the embeddings: BM25, and a fusion of the two rankings.

Why a second retriever at all. Ten of the twenty-six answerable questions were
not served their gold span in the top five, and the misses were not random:
they cluster on questions that name a document by its identifier - "What is the
scope of NIST SP 800-78-5?", "What flexibility does SP 800-171Ar3 give to
assessors?". Dense retrieval ranks those gold chunks below position fifty.

**The reasoning that followed from that was wrong, and it is left here because
the fix it produced works anyway.** The argument was that an identifier is an
exact string and exact strings are what word matching does better. Measured on
17.09 (docs/lessons.md #29): neither gold span contains the identifier at all -
one says "The assessment procedures are flexible and can be customized", the
other "This document contains the technical specifications needed for" - so
there is no exact string in the text to match. BM25 ranks the first of the two
at 30 and never finds the second inside 300; the fused top five contains
neither at any depth tried.

What the fusion did improve is four other records, and it cost two. That is a
real gain and it is not the gain this paragraph predicted.

The tokenizer has no stopword list, which is the other half of it: in formal
NIST prose the question words are rare and therefore heavy. "does" scores
idf 4.76 and "what" 4.57, against 3.54 for "800-171ar3" - so for the scope
question BM25's top three come from two other documents entirely, ranked on
"what" and "scope". Fixing that is a retrieval change and would move every
number in the README, so it is written down rather than done.

BM25 is the standard answer and it is twenty lines. Terms are scored by how
often they appear in a chunk (saturating, so a chunk that repeats one word
does not win on that alone), damped by how common the term is across the
corpus, and normalised by chunk length so long chunks do not collect matches
for free.

    k1  how quickly term frequency saturates. 1.5 is the usual choice.
    b   how strongly to normalise by length. 0.75 is the usual choice.

Both are left at the values everyone uses. They are worth tuning only against a
measurement, and the measurement here is 26 questions - far too few to fit two
parameters without fitting the fixture instead.

Fusing the two rankings uses Reciprocal Rank Fusion rather than a weighted sum
of scores. A BM25 score and a cosine distance are not on one scale, and any
formula that adds them needs a weight that has to be tuned - again, on 26
questions. RRF only reads positions: a chunk ranked r by a retriever
contributes 1/(RRF_K + r), and a chunk both retrievers rank highly beats one
that either ranks first alone. It has no free parameter worth fitting; RRF_K=60
is the value from the paper and everyone's default.
"""

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

# Dampens the top of the ranking so that rank 1 is not overwhelmingly better
# than rank 2. 60 comes from the original RRF paper.
RRF_K = 60

# Keeps digits and hyphens together, so "800-78-5" survives as one term instead
# of becoming three numbers. That is the whole point of having this retriever.
# A hyphen or dot only counts inside the token, never at the end: written as
# [a-z0-9][a-z0-9.-]* it also kept the full stop after a sentence, and
# "monitor." and "monitor" then indexed as two unrelated terms.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class BM25:
    """BM25 over a fixed list of chunks, built once and queried many times."""

    def __init__(self, chunks, k1=K1, b=B):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(chunk["text"]) for chunk in chunks]
        self.lengths = [len(doc) for doc in self.docs]
        self.average = sum(self.lengths) / len(self.docs) if self.docs else 0.0
        self.frequencies = [Counter(doc) for doc in self.docs]

        containing = Counter()
        for doc in self.docs:
            containing.update(set(doc))
        total = len(self.docs)
        # The +0.5 smoothing is the standard form: it keeps the idf of a term
        # that appears in every chunk from going negative and quietly turning
        # a match into a penalty.
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in containing.items()
        }

    def scores(self, query):
        terms = [term for term in tokenize(query) if term in self.idf]
        out = [0.0] * len(self.docs)
        for i, frequencies in enumerate(self.frequencies):
            length = self.lengths[i] or 1
            total = 0.0
            for term in terms:
                count = frequencies.get(term, 0)
                if not count:
                    continue
                norm = count + self.k1 * (1 - self.b + self.b * length / self.average)
                total += self.idf[term] * count * (self.k1 + 1) / norm
            out[i] = total
        return out

    def search(self, query, k=5):
        scored = sorted(enumerate(self.scores(query)), key=lambda pair: -pair[1])
        return [
            {**self.chunks[i], "score": score}
            for i, score in scored[:k]
            if score > 0
        ]


def rrf(rankings, k=5, rrf_k=RRF_K):
    """
    Fuse several ranked lists of chunks by Reciprocal Rank Fusion.

    rankings is a list of ranked lists; each entry is a chunk dict carrying at
    least "text". Chunks are identified by their text, which is what both
    retrievers return and what the metrics compare against.
    """
    fused = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            key = chunk["text"]
            entry = fused.setdefault(key, {"chunk": chunk, "score": 0.0, "ranks": []})
            entry["score"] += 1 / (rrf_k + rank)
            entry["ranks"].append(rank)
    ordered = sorted(fused.values(), key=lambda entry: -entry["score"])
    return [
        {**entry["chunk"], "rrf_score": entry["score"], "ranks": entry["ranks"]}
        for entry in ordered[:k]
    ]
