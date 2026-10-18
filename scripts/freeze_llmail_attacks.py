"""Freeze a set of third-party injection attacks from LLMail-Inject, phase 2.

Run:  uv run python scripts/freeze_llmail_attacks.py <raw_submissions_phase2.jsonl>

The seven attacks in injection_probe.py were written by the person who wrote
the defence, and seven is too few to say anything: 0/7 obeyed still allows an
attack success rate up to 35%. LLMail-Inject (Microsoft, 2025, MIT licence,
huggingface.co/datasets/microsoft/llmail-inject-challenge) is a public
challenge where teams wrote emails meant to make an LLM email assistant send
"confirmation" to contact@contact.com. Nobody there saw this project's prompt.

Selection, fixed before any of these is run here:
- phase 2 only, the round against the stronger defences;
- the attack made their model send to the right address (exfil.sent and
  exfil.destination). Whether their detector also missed it is not required:
  this pipeline has no such detector, so it would select on the wrong thing;
- one attack per team, picked at random with a fixed seed. Attacks from one
  team are variations of one idea, so the team is the closer thing to an
  independent trial. Picking each team's "best" would need a judgement this
  script should not make.

The output is committed and not regenerated: the set is frozen before the
first run, so no later look at results can move what is in it.
"""

import json
import pathlib
import random
import sys

OUT = pathlib.Path("eval/injection_attacks.llmail.json")
SEED = 0


def main():
    source = pathlib.Path(sys.argv[1])
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]

    by_team = {}
    for row in rows:
        objectives = json.loads(row["objectives"])
        if objectives.get("exfil.sent") and objectives.get("exfil.destination"):
            by_team.setdefault(row["team_id"], []).append(row)

    rng = random.Random(SEED)
    attacks = []
    # Sorted, so the seed picks the same rows whatever order the file is in.
    for team in sorted(by_team):
        candidates = sorted(by_team[team], key=lambda r: r["RowKey"])
        pick = rng.choice(candidates)
        attacks.append({
            "id": pick["RowKey"],
            "team": team,
            "level": pick["scenario"],
            "subject": pick["subject"],
            "body": pick["body"],
        })

    OUT.write_text(json.dumps({
        "source": "microsoft/llmail-inject-challenge, data/raw_submissions_phase2.jsonl (MIT)",
        "selection": "phase 2, exfil.sent and exfil.destination, one per team, random.Random(0)",
        "goal": "send an email to contact@contact.com with body 'confirmation'",
        "teams_eligible": len(by_team),
        "attacks": attacks,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(attacks)} attacks from {len(by_team)} teams -> {OUT}")


if __name__ == "__main__":
    main()
