"""
Precompute BERT masked LM scores for each word in a sentence.
Mirrors the protein precompute script — masks each word, gets log-probs
for a fixed vocabulary of substitutions, saves to nlp_scores.json.

Install:
    pip install transformers torch

Usage:
    python3 precompute_nlp.py

Output:
    nlp_scores.json — place in same folder as mutation-explorer.html
"""

import json, math, torch
from transformers import BertTokenizer, BertForMaskedLM

# ── config ────────────────────────────────────────────────────────────────────
SENTENCE = "The scientist reads each paper and adds one new idea at a time while thinking"
# 15 words, no period (BERT handles it fine either way)

# For each word position, we score these candidate substitutions.
# Chosen to cover: synonyms, wrong-POS, absurd-but-grammatical, nonsense.
# Keep to ~8-12 per position so the widget isn't overwhelming.
CANDIDATES = {
    0:  ["the", "a", "this", "every", "each", "one", "that", "any", "some", "no", "our", "their", "purple", "quickly", "glorp"],
    1:  ["scientist", "researcher", "biologist", "student", "professor", "doctor", "engineer", "ribosome", "cat", "machine", "robot", "banana", "17", "loudly", "zorp"],
    2:  ["reads", "studies", "analyzes", "examines", "processes", "writes", "skips", "eats", "runs", "sleeps", "blue", "quickly", "table", "glorp", "zorp"],
    3:  ["each", "every", "one", "many", "all", "another", "any", "some", "no", "two", "three", "several", "purple", "loudly", "blarg"],
    4:  ["paper", "article", "book", "study", "report", "journal", "document", "chapter", "thesis", "text", "banana", "cat", "idea", "zork", "glorp"],
    5:  ["and", "then", "but", "or", "while", "yet", "so", "nor", "before", "after", "because", "although", "banana", "splat", "zorp"],
    6:  ["adds", "includes", "contributes", "provides", "introduces", "removes", "creates", "writes", "skips", "loudly", "blue", "table", "banana", "zorp", "glorp"],
    7:  ["one", "a", "another", "each", "two", "three", "some", "any", "no", "every", "that", "this", "zero", "purple", "zorp"],
    8:  ["new", "fresh", "novel", "original", "unique", "important", "interesting", "surprising", "old", "strange", "weird", "broken", "quickly", "purple", "blorg"],
    9:  ["idea", "concept", "thought", "insight", "theory", "finding", "question", "hypothesis", "solution", "discovery", "approach", "banana", "cat", "7", "zork"],
    10: ["at", "per", "during", "after", "before", "within", "inside", "beyond", "through", "across", "around", "toward", "over", "banana", "zork"],
    11: ["a", "the", "each", "one", "every", "another", "this", "that", "any", "no", "some", "their", "our", "purple", "zorp"],
    12: ["time", "step", "turn", "moment", "point", "stage", "iteration", "pass", "cycle", "round", "attempt", "try", "banana", "cat", "zork"],
    13: ["while", "as", "when", "because", "although", "though", "since", "before", "after", "if", "unless", "until", "and", "banana", "blarg"],
    14: ["thinking", "working", "reading", "writing", "learning", "studying", "sleeping", "eating", "running", "walking", "sitting", "waiting", "dancing", "loudly", "zorp"],
}

# ── load BERT ─────────────────────────────────────────────────────────────────
print("Loading BERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()
print("Done.\n")

words = SENTENCE.split()
assert len(words) == 15, f"Expected 15 words, got {len(words)}"
print(f"Sentence: {SENTENCE}")
print(f"Words:    {words}\n")

scores = {}

with torch.no_grad():
    for pos, word in enumerate(words):
        # build masked sentence
        masked_words   = words[:pos] + ["[MASK]"] + words[pos+1:]
        masked_sentence = " ".join(masked_words)

        # tokenize — BERT adds [CLS] and [SEP]
        encoding   = tokenizer(masked_sentence, return_tensors="pt")
        input_ids  = encoding["input_ids"]
        tokens     = tokenizer.convert_ids_to_tokens(input_ids[0])

        # find which token index is [MASK]
        mask_idx   = tokens.index("[MASK]")

        # run model
        output     = model(**encoding)
        logits     = output.logits[0, mask_idx]  # (vocab_size,)
        log_probs  = torch.log_softmax(logits, dim=-1)

        # score each candidate word
        # note: BERT is uncased so lowercase everything
        cands      = CANDIDATES[pos]
        pos_scores = {}
        for cand in cands:
            cand_ids = tokenizer.encode(cand.lower(), add_special_tokens=False)
            if len(cand_ids) == 1:
                lp = log_probs[cand_ids[0]].item()
            else:
                # multi-token word — use mean log-prob as approximation
                lp = sum(log_probs[i].item() for i in cand_ids) / len(cand_ids)
            pos_scores[cand] = round(lp, 4)

        # compute delta relative to wildtype
        wt_score  = pos_scores[word.lower()]
        best_alt  = max(s for w, s in pos_scores.items() if w != word.lower())
        delta     = best_alt - wt_score
        print(f"  [{pos+1:2d}/15] {word:12s}  WT: {wt_score:.3f}  best alt: {best_alt:.3f}  Δ: {delta:+.3f}")

        scores[str(pos)] = pos_scores

# ── save ──────────────────────────────────────────────────────────────────────
output = {
    "sentence": SENTENCE,
    "words":    words,
    "model":    "bert-base-uncased",
    "scores":   scores,
}
with open("nlp_scores.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\n✓ Saved nlp_scores.json")

# ── delta distribution ────────────────────────────────────────────────────────
all_deltas = sorted(
    s - scores[str(pos)][words[pos].lower()]
    for pos in range(15)
    for w, s in scores[str(pos)].items()
    if w != words[pos].lower()
)
n = len(all_deltas)
print(f"\nDelta distribution:")
print(f"  best:   {all_deltas[-1]:+.3f}")
print(f"  75th%:  {all_deltas[3*n//4]:+.3f}")
print(f"  median: {all_deltas[n//2]:+.3f}")
print(f"  worst:  {all_deltas[0]:+.3f}")
