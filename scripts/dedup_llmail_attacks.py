"""Replace near-duplicate attacks in the frozen LLMail set, next to it, not over it.

Run:  uv run python scripts/dedup_llmail_attacks.py <raw_submissions_phase2.jsonl>

The frozen set is one attack per team, but teams shared emails: 6de2a377 and
b12d836d differ by one character, and five teams sent variations of one
"good vibes, coffee next week" template. Counting them as five trials
counts one idea five times.

Rule, fixed before any replacement is run here:
- similarity is difflib's ratio on subject + body; two attacks are one
  family at 0.75 or more, and families are connected components (A~B and
  B~C put A and C together even when A and C are further apart);
- each family keeps its first member in the frozen order (teams sorted by
  id), whatever any probe said about it;
- every other member is replaced by another eligible attack from the same
  team, drawn with random.Random(0) over the team's rows sorted by RowKey,
  the first that is below 0.75 to everything in the frozen set and to the
  replacements already drawn;
- a team with no such attack drops out, and the set gets smaller.

eval/injection_attacks.llmail.json stays as it is; the results measured on it
stay comparable to each other.
"""

import difflib
import itertools
import json
import pathlib
import random
import sys

FROZEN = pathlib.Path("eval/injection_attacks.llmail.json")
OUT = pathlib.Path("eval/injection_attacks.llmail.dedup.json")
THRESHOLD = 0.75
SEED = 0


def text(attack):
    return f"{attack['subject']}\n{attack['body']}"


def similar(a, b):
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    # The quick ratios are upper bounds: most pairs are rejected without the full one.
    return (m.real_quick_ratio() >= THRESHOLD and m.quick_ratio() >= THRESHOLD
            and m.ratio() >= THRESHOLD)


def families(attacks):
    """Connected components of the similarity graph, as lists of indices in frozen order."""
    parent = list(range(len(attacks)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in itertools.combinations(range(len(attacks)), 2):
        if similar(text(attacks[i]), text(attacks[j])):
            parent[root(j)] = root(i)
    groups = {}
    for i in range(len(attacks)):
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def main():
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    attacks = frozen["attacks"]

    by_team = {}
    for line in pathlib.Path(sys.argv[1]).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        objectives = json.loads(row["objectives"])
        if objectives.get("exfil.sent") and objectives.get("exfil.destination"):
            by_team.setdefault(row["team_id"], []).append(row)

    taken = [text(a) for a in attacks]
    rng = random.Random(SEED)
    replaced, dropped = {}, []
    for family in families(attacks):
        print("family:", " ".join(attacks[i]["id"][:8] for i in family))
        for i in family[1:]:
            old = attacks[i]
            candidates = sorted(by_team[old["team"]], key=lambda r: r["RowKey"])
            rng.shuffle(candidates)
            pick = next((r for r in candidates
                         if not any(similar(f"{r['subject']}\n{r['body']}", t) for t in taken)), None)
            if pick is None:
                dropped.append(old["id"])
                print(f"  {old['id'][:8]} dropped: {len(candidates)} eligible, all in a family")
                continue
            new = {"id": pick["RowKey"], "team": old["team"], "level": pick["scenario"],
                   "subject": pick["subject"], "body": pick["body"], "replaces": old["id"]}
            replaced[old["id"]] = new
            taken.append(text(new))
            print(f"  {old['id'][:8]} -> {new['id'][:8]} ({len(candidates)} eligible)")

    out = [replaced.get(a["id"], a) for a in attacks if a["id"] not in dropped]
    OUT.write_text(json.dumps({
        **{k: v for k, v in frozen.items() if k != "attacks"},
        "dedup": f"families at difflib ratio >= {THRESHOLD}, first kept, the rest redrawn "
                 f"from the same team with random.Random({SEED}); see scripts/dedup_llmail_attacks.py",
        "replaced": len(replaced),
        "dropped": dropped,
        "attacks": out,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(out)} attacks ({len(replaced)} replaced, {len(dropped)} dropped) -> {OUT}")


if __name__ == "__main__":
    main()
