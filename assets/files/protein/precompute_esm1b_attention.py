"""
Run this in an environment with internet access to HuggingFace / fair-esm and a GPU
(or patience — ESM-1b is 650M params, fine on CPU for one 76-residue sequence).

pip install fair-esm torch

Produces attention_export.json in exactly the schema esm1b_attention_explorer.html
expects for ATTENTION_HEADS. Paste that array in to replace the placeholder.
"""

import json
import torch
import esm

UBIQUITIN_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# Pick whichever heads you want to expose in the dropdown. ESM-1b has 33 layers x 20 heads.
# A reasonable starting spread: a couple of early, mid, and late layers.
HEADS_TO_EXPORT = [
    (12, 3), (12, 7), (4, 11), (4, 2), (8, 15), (20, 6),
]

def main():
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    data = [("ubiquitin", UBIQUITIN_SEQ)]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        out = model(tokens, repr_layers=[], need_head_weights=True)

    # attentions shape: [batch, layers, heads, seq_len, seq_len] (includes BOS/EOS tokens)
    attentions = out["attentions"][0]  # drop batch dim -> [layers, heads, L+2, L+2]

    n = len(UBIQUITIN_SEQ)
    export = []
    for layer, head in HEADS_TO_EXPORT:
        # layer is 1-indexed in common usage; attentions is 0-indexed by layer
        mat = attentions[layer - 1, head - 1, 1:n+1, 1:n+1]  # strip BOS/EOS, keep real residues
        export.append({
            "layer": layer,
            "head": head,
            "label": f"Layer {layer} · Head {head}",
            "matrix": mat.tolist(),
        })

    with open("attention_export.json", "w") as f:
        json.dump(export, f)

    print(f"Wrote attention_export.json with {len(export)} heads over {n} residues.")
    print("Paste this array into ATTENTION_HEADS in esm1b_attention_explorer.html")
    print("(mapping label -> label, matrix -> matrix; buildHead(...) calls get replaced directly by the real matrix).")

if __name__ == "__main__":
    main()
