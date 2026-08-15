"""
Precompute ESM mutation scores using BioLM's predictor endpoint.
Scores the full ubiquitin sequence for context, but only exports
the first 15 positions to mutation_scores.json for the widget.

Install:
    pip install requests

Usage:
    BIOLM_KEY=your_key_here python3 precompute.py

Get your key at: https://biolm.ai
"""

import os, json, math, time, requests

# ── config ────────────────────────────────────────────────────────────────────
KEY          = os.environ.get("BIOLM_KEY", "YOUR_KEY_HERE")
SEQUENCE     = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
DISPLAY_SEQ  = SEQUENCE[:15]
EXPORT_RANGE = range(15)
URL          = "https://biolm.ai/api/v3/esmc-300m/predict/"
HEADERS      = {"Authorization": f"Token {KEY}", "Content-Type": "application/json"}

print(f"Full sequence ({len(SEQUENCE)} residues): {SEQUENCE}")
print(f"Scoring all {len(SEQUENCE)} positions for context, exporting first {len(DISPLAY_SEQ)}...")
print(f"Estimated time: ~{len(SEQUENCE) * 2}s\n")

scores = {}

for pos in range(len(SEQUENCE)):
    wt     = SEQUENCE[pos]
    masked = SEQUENCE[:pos] + "<mask>" + SEQUENCE[pos+1:]

    print(f"  [{pos+1:2d}/{len(SEQUENCE)}] {wt}", end=" ", flush=True)

    try:
        resp = requests.post(URL, headers=HEADERS,
                             json={"items": [{"sequence": masked}]},
                             timeout=30)
        resp.raise_for_status()
        result = resp.json()["results"][0]

        vocab      = result["vocab_tokens"]
        seq_tokens = result["sequence_tokens"]
        masked_idx = seq_tokens.index("_")
        logits     = result["logits"][masked_idx]

        max_l   = max(logits)
        exp_sum = sum(math.exp(v - max_l) for v in logits)
        log_z   = math.log(exp_sum)

        pos_scores = {aa: round((logits[i] - max_l - log_z), 4)
                      for i, aa in enumerate(vocab)
                      if aa in "ACDEFGHIKLMNPQRSTVWY"}

        wt_score = pos_scores.get(wt, 0)
        best_alt = max((s for a, s in pos_scores.items() if a != wt), default=0)
        delta    = best_alt - wt_score
        marker   = " ◀" if pos in EXPORT_RANGE else ""
        print(f"WT:{wt_score:.2f} best:{best_alt:.2f} Δ:{delta:+.2f}{marker}")

        if pos in EXPORT_RANGE:
            scores[str(pos)] = pos_scores

    except Exception as e:
        print(f"ERROR: {e}")

    time.sleep(0.25)

# ── save ──────────────────────────────────────────────────────────────────────
output = {
    "sequence": DISPLAY_SEQ,
    "model":    "esmc-300m (BioLM, full ubiquitin context)",
    "scores":   scores,
}
with open("mutation_scores.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved mutation_scores.json ({len(scores)} positions)")

# ── threshold suggestions ─────────────────────────────────────────────────────
all_deltas = sorted(
    s - scores[str(pos)][SEQUENCE[pos]]
    for pos in EXPORT_RANGE
    for aa, s in scores[str(pos)].items()
    if aa != SEQUENCE[pos]
)
n = len(all_deltas)
print(f"\nDelta distribution (exported positions):")
print(f"  best:   {all_deltas[-1]:+.3f}")
print(f"  75th%:  {all_deltas[3*n//4]:+.3f}")
print(f"  median: {all_deltas[n//2]:+.3f}")
print(f"  worst:  {all_deltas[0]:+.3f}")
print(f"\nSuggested thresholds for mutation-explorer.html:")
print(f"  tolerated  > {all_deltas[3*n//4]:.2f}")
print(f"  harmful    {all_deltas[n//2]:.2f} to {all_deltas[3*n//4]:.2f}")
print(f"  disruptive < {all_deltas[n//2]:.2f}")
