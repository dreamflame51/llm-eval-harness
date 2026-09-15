"""Compare RAGAS, DeepEval and this harness's own metrics on the same answers.

Run:  uv run python scripts/compare_evals.py

Calls no model. Everything here is read back from eval/ragas_scores.json and
eval/deepeval_scores.json, both committed, both computed over the same frozen
answers in eval/answerable_answers.yaml.

Running two libraries over one set of answers turns a library choice into a
measurement. The comparison is worth more than either score alone, because the
two agree on what to call the metrics and not on what to compute:

  the means can agree while the records do not. Averaged over 26 answers the
  two faithfulness scores sit within 0.1 of each other. Per record they
  correlate at 0.11. An aggregate that matches is not evidence that two
  implementations measure the same thing.

  an answer with no claims in it has no agreed score. "Answer is missing"
  scores faithfulness 1.00 in DeepEval on all six records where the model
  declined - nothing in it is contradicted - and 0.00, 0.50 or 1.00 in RAGAS
  on identical text, depending on whether its claim extractor found anything
  in four words to check. A system that declined every question would be
  perfectly faithful to one library and unpredictable to the other.

  a judge in a library is still a judge. One DeepEval verdict here says a
  document title is "not mentioned in the retrieval context" when it is in
  chunk 4, verbatim. The reason field is what makes that checkable, which is
  the same argument as the evidence quote in judge.py.

The harness's own retrieval numbers are printed alongside where they answer a
comparable question, with the caveat that they are not the same measurement:
coverage@5 counts word runs from the gold span, context_recall asks a model
whether the reference is covered.
"""

import json
import pathlib
import statistics

import yaml

from llm_eval_harness.refusal import looks_like_refusal

ANSWERS = pathlib.Path("eval/answerable_answers.yaml")
RAGAS = pathlib.Path("eval/ragas_scores.json")
DEEPEVAL = pathlib.Path("eval/deepeval_scores.json")

# The same three files as they were before BM25 was fused into retrieval. Kept
# because a metric moving is only readable next to what it moved from, and
# because re-running them costs an hour and a half of inference.
BASELINE = pathlib.Path("eval/dense_baseline")

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load():
    """
    The answers and whatever scores exist.

    A missing score file is an ordinary state, not an error: the two runs cost
    an hour and several hours, they are run separately, and the half that
    exists is still worth reading. Sections that need the other half say so
    and are skipped.
    """
    rows = yaml.safe_load(ANSWERS.read_text(encoding="utf-8"))
    ragas = (
        json.loads(RAGAS.read_text(encoding="utf-8"))["per_record"]
        if RAGAS.exists()
        else {}
    )
    deepeval = (
        {
            row["question"]: row
            for row in json.loads(DEEPEVAL.read_text(encoding="utf-8"))["per_record"]
        }
        if DEEPEVAL.exists()
        else {}
    )
    return rows, ragas, deepeval


def paired(ragas, deepeval, metric):
    """(ragas, deepeval) for every record both libraries scored."""
    pairs = []
    for question, scores in ragas.items():
        other = deepeval.get(question, {})
        if scores.get(metric) is not None and other.get(metric) is not None:
            pairs.append((scores[metric], other[metric]))
    return pairs


def correlation(pairs):
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(left, right)


def baseline_means():
    """{library: {metric: mean}} from the pre-hybrid run, or {} if absent."""
    out = {}
    for name, path in (
        ("RAGAS", BASELINE / "ragas_scores.json"),
        ("DeepEval", BASELINE / "deepeval_scores.json"),
    ):
        if not path.exists():
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))["per_record"]
        records = stored.values() if isinstance(stored, dict) else stored
        rows = list(records)
        out[name] = {
            metric: statistics.mean(
                [row[metric] for row in rows if row.get(metric) is not None]
            )
            for metric in METRICS
            if any(row.get(metric) is not None for row in rows)
        }
    return out


def print_against_baseline(ragas, deepeval):
    before = baseline_means()
    if not before:
        return
    now = {}
    for library, scores in (("RAGAS", ragas), ("DeepEval", deepeval)):
        now[library] = {
            metric: statistics.mean(values)
            for metric in METRICS
            if (values := [r[metric] for r in scores.values() if r.get(metric) is not None])
        }
    print("\ndense retrieval -> hybrid (BM25 fused in), same questions")
    for library in ("RAGAS", "DeepEval"):
        if library not in before or not now.get(library):
            continue
        print(f"  {library}")
        for metric in METRICS:
            if metric not in before[library] or metric not in now[library]:
                continue
            was, is_now = before[library][metric], now[library][metric]
            print(f"    {metric:<18}{was:>7.3f} -> {is_now:>6.3f}  {is_now - was:+.3f}")
    print(
        "  Retrieval changed, so the answers changed and these are different\n"
        "  answers scored by the same metric - not the same answers rescored."
    )


def main():
    rows, ragas, deepeval = load()
    refusals = [row for row in rows if looks_like_refusal(row["answer"])]

    print(f"{len(rows)} answerable questions, one frozen answer each\n")

    if not ragas or not deepeval:
        missing = "RAGAS" if not ragas else "DeepEval"
        print(f"no {missing} scores yet - the side-by-side needs both\n")

    header = f"{'metric':<18}{'RAGAS':>8}{'DeepEval':>10}{'corr':>7}{'far apart':>11}"
    print(header)
    print("-" * len(header))
    for metric in METRICS:
        pairs = paired(ragas, deepeval, metric)
        if not pairs:
            continue
        corr = correlation(pairs)
        far = sum(1 for a, b in pairs if abs(a - b) > 0.5)
        print(
            f"{metric:<18}"
            f"{statistics.mean(a for a, _ in pairs):>8.3f}"
            f"{statistics.mean(b for _, b in pairs):>10.3f}"
            f"{'    n/a' if corr is None else f'{corr:>7.2f}'}"
            f"{far:>11}"
        )
    print("\ncorr is per record. far apart counts records differing by more than 0.5.")

    print_against_baseline(ragas, deepeval)

    # The finding neither library was run to produce, and the one the harness
    # could not have seen: refusal.py only scores the unanswerable records and
    # evaluator.py only scores retrieval, so an answerable question the model
    # declined fell between them.
    print(f"\ndeclined an answerable question: {len(refusals)} of {len(rows)}")
    for row in refusals:
        print(f"  {row['question'][:70]}")

    if refusals:
        print("\nhow each library scores a declined answer:")
        print(f"  {'':<44}{'faithfulness':>16}{'answer_relevancy':>20}")
        print(f"  {'answer':<44}{'RAGAS  DeepEval':>16}{'RAGAS  DeepEval':>20}")
        for row in refusals:
            scores = ragas.get(row["question"], {})
            other = deepeval.get(row["question"], {})
            print(
                f"  {row['answer'][:42]:<44}"
                f"{_num(scores.get('faithfulness')):>8}{_num(other.get('faithfulness')):>8}"
                f"{_num(scores.get('answer_relevancy')):>12}"
                f"{_num(other.get('answer_relevancy')):>8}"
            )
        print(
            "\n  Three things in that block, none of them about the system:\n"
            "  - DeepEval calls every declined answer perfectly faithful. Nothing in\n"
            "    'Answer is missing' contradicts anything, so it scores 1.00, and a\n"
            "    system that declined every question would score a perfect 1.00.\n"
            "  - RAGAS is not stable on the same input: the identical answer scores\n"
            "    0.00, 0.50 and 1.00 across these records, depending on whether its\n"
            "    claim extractor found a claim to check in four words.\n"
            "  - Only RAGAS's answer_relevancy is consistent here: 0.00 on all six,\n"
            "    which is the defensible reading - a decline does not address the\n"
            "    question. DeepEval's reads four of the six as fully relevant.\n"
            "  Read the definition, not the number."
        )

    print("\nwhere the two libraries disagree most")
    gaps = []
    for question, scores in ragas.items():
        other = deepeval.get(question, {})
        for metric in METRICS:
            a, b = scores.get(metric), other.get(metric)
            if a is not None and b is not None and abs(a - b) > 0.6:
                gaps.append((abs(a - b), metric, a, b, question, other))
    for _, metric, a, b, question, other in sorted(gaps, key=lambda g: -g[0])[:6]:
        print(f"  {metric:<18} RAGAS {a:.2f}  DeepEval {b:.2f}  {question[:52]}")
        reason = (other.get(f"{metric}_reason") or "").strip()
        if reason:
            print(f"    DeepEval: {reason[:150]}")

    print(
        "\nThe harness's own retrieval numbers over the same fixture: "
        "hit@5 0.423, covered@5 0.462,\ncoverage@5 0.544 "
        "(uv run python -m llm_eval_harness.evaluator). They are computed from "
        "word runs\nagainst the gold span, not by a model, and they answer a "
        "narrower question than\ncontext_recall does - compare the direction, "
        "not the digits."
    )


def _num(value):
    return "  -  " if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
