#!/usr/bin/env python3
"""Replay saved bulk-labeling decisions through review gates. No API calls.

python3 confidence_gating.py --input labeling_results.jsonl
Uses only Python's standard library; no .env or API keys needed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

LABELS = {"billing", "login", "technical_issue", "other"}
THRESHOLDS = (0.8, 0.95, 1.0)


def valid_confidence(value):
    return (type(value) in (int, float) and math.isfinite(value)
            and 0 <= value <= 1)


def gate(row, threshold=None):
    """None means label-only gating, not zero confidence."""
    if row.get("status") != "ok" or row.get("label") not in LABELS:
        return "review", "failed_or_invalid"
    if row["label"] == "other":
        return "review", "other_label"
    if threshold is not None:
        score = row.get("confidence")
        if not valid_confidence(score):
            return "review", "missing_or_invalid_confidence"
        if score < threshold:
            return "review", "below_threshold"
    return "automatic", "accepted"


def load_records(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Input is empty")
    seen = set()
    for row in records:
        if row.get("provider") not in {"jev", "openai"}:
            raise ValueError("Unknown provider")
        if row.get("reference_label") not in LABELS:
            raise ValueError("Reference label missing or invalid")
        key = (row["provider"], row["case"], row["repeat"])
        if key in seen:
            raise ValueError("Duplicate provider/case/repeat")
        seen.add(key)
        if row.get("status") == "ok":
            if row.get("label") not in LABELS:
                raise ValueError("Unknown successful label")
            # Cross-check saved values against the original provider response.
            raw = row["raw_response"]
            if row["provider"] == "jev":
                answer = raw["answers"]["label"]
                if (answer["choice"] != row["label"] or
                    answer["confidence"] != row.get("confidence") or
                    not valid_confidence(row.get("confidence"))):
                    raise ValueError("Inconsistent Jev result")
            else:
                answer = json.loads(raw["choices"][0]["message"]["content"])
                if answer["label"] != row["label"]:
                    raise ValueError("Inconsistent OpenAI result")
    return records


def analyze(records):
    summaries, decisions = [], []
    policies = [("jev", None), *[("jev", t) for t in THRESHOLDS], ("openai", None)]
    for provider, threshold in policies:
        rows = [r for r in records if r["provider"] == provider]
        if not rows:
            continue
        name = "other_to_review" if threshold is None else f"other_to_review_confidence_ge_{threshold:g}"
        routed = [(r, *gate(r, threshold)) for r in rows]
        automatic = [r for r, outcome, reason in routed if outcome == "automatic"]
        wrong = sum(r["label"] != r["reference_label"] for r in automatic)
        summaries.append({
            "provider": provider, "policy": name, "threshold": threshold,
            "attempted_records": len(rows), "distinct_cases": len({r["case"] for r in rows}),
            "valid_labels": sum(r.get("status") == "ok" for r in rows),
            "automatic": len(automatic), "review": len(rows)-len(automatic),
            "automatic_coverage": len(automatic)/len(rows),
            "automatic_errors": wrong,
            "automatic_error_rate": wrong/len(automatic) if automatic else None,
            "review_reasons": dict(Counter(reason for r, outcome, reason in routed if outcome == "review")),
        })
        decisions.extend({"provider":provider,"policy":name,"case":r["case"],
                          "repeat":r["repeat"],"label":r.get("label"),
                          "outcome":outcome,"reason":reason}
                         for r, outcome, reason in routed)
    return summaries, decisions


def self_test():
    base = {"status":"ok","label":"billing","confidence":.8}
    assert gate(base,.8)[0] == "automatic"
    assert gate(base,.81)[0] == "review"
    assert gate({**base,"label":"other","confidence":1},.8)[1] == "other_label"
    for confidence in (None,True,float("nan"),1.1):
        assert gate({**base,"confidence":confidence},.8)[0] == "review"
    assert gate({"status":"error"},.8)[0] == "review"
    assert gate({"status":"ok","label":"billing"},None)[0] == "automatic"
    summary,_=analyze([{"provider":"jev","case":"x","repeat":1,
                        "reference_label":"other",**base}])
    assert summary[0]["automatic_errors"]==1
    assert summary[-1]["automatic_error_rate"] is None
    print("Offline checks passed: threshold boundary, review overrides, missing confidence, failures, and zero-coverage metrics.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("labeling_results.jsonl"))
    parser.add_argument("--out",type=Path,default=Path("confidence_gating_results"))
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    records=load_records(args.input)
    summary,decisions=analyze(records)
    args.out.mkdir(parents=True,exist_ok=True)
    report={"source_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "additional_api_calls":0,"additional_api_cost_usd":0,
            "notes":"Replay of attempted records only; thresholds exploratory, not calibrated. OpenAI baseline has no confidence gate. Review outcomes not adjudicated.",
            "policies":summary}
    (args.out/"confidence_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.out/"confidence_decisions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in decisions))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
