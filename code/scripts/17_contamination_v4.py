"""
Contamination / near-duplicate audit of the HARD pool before decontamination (N=883).

Embeds each HARD problem and each reference-benchmark problem with
sentence-transformers/all-mpnet-base-v2 (768-d, normalized) and reports, per
reference corpus, the max cosine similarity and the count of HARD problems with a
match >= 0.85 (near-duplicate) and >= 0.95 (near-literal). It also writes the full
per-problem match table (best match in every reference corpus).

Local only (no paid API). Model and reference corpora are downloaded once.

Output: results/contamination_v4_report.json and results/contamination_pairs.json
"""
from __future__ import annotations
import importlib.util, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
(ROOT / "results").mkdir(exist_ok=True)
import numpy as np

# --- the audited pool: HARD after the guard, before decontamination (200 + 683) ---
spec = importlib.util.spec_from_file_location("rs", Path(__file__).parent / "19_review_support.py")
rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
rs.DROP = set()
audited = []
for stage, src in (("stage_a", "stage_a_1000.json"), ("stage_b", "stage_b_3000.json")):
    probs = {p["post_neurips_id"]: p for p in json.load(open(DATA / src, encoding="utf-8"))}
    _, _, _, _, hard_guard = rs.build(stage, rs.JUDGES_ALL, 2)
    audited += [(stage, pid, probs[pid]["problem"]) for pid in hard_guard]
hard_texts = [t for _, _, t in audited]
print(f"HARD problems loaded: {len(hard_texts)}")

# --- reference corpora ---
refs = {}

# AIME (local, essential)
aime_local = DATA / "aime" / "aime_problems.json"
if aime_local.exists():
    aime = [p["problem"] for p in json.load(open(aime_local, encoding="utf-8"))]
else:
    from datasets import load_dataset as _ld
    aime = [r["Problem"] for r in _ld("gneubig/aime-1983-2024", split="train")]
refs["AIME 1983-2024"] = [t for t in aime if t]

# best-effort HF loads
def try_hf(name, loader):
    try:
        texts = loader()
        texts = [t for t in texts if t and len(t) > 10]
        if texts:
            refs[name] = texts
            print(f"  loaded {name}: {len(texts)}")
    except Exception as e:
        print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:120]}")

from datasets import load_dataset

try_hf("MATH-500", lambda: [r["problem"] for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
try_hf("Putnam-AXIOM", lambda: [
    (r.get("problem") or r.get("question") or r.get("problem_statement") or "")
    for r in load_dataset("Putnam-AXIOM/putnam-axiom-original", split="func")
])
try_hf("OlympicArena-Math", lambda: [
    (r.get("problem") or r.get("question") or "")
    for r in load_dataset("GAIR/OlympicArena", "Math", split="val")
])

print(f"Reference corpora: { {k: len(v) for k, v in refs.items()} }")

# --- embed ---
from sentence_transformers import SentenceTransformer
print("Loading embedder all-mpnet-base-v2 (one-time download if needed)...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def embed(texts):
    return model.encode(texts, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)

H = embed(hard_texts)  # (883, 768)

report = {"n_hard": len(hard_texts), "per_dataset": {}}
pairs = [{"pid": pid, "stage": st, "best_match": {}} for st, pid, _ in audited]
union_near = set()
union_lit = set()
for name, texts in refs.items():
    R = embed(texts)               # (Nref, 768)
    sims = H @ R.T                  # (883, Nref) cosine (normalized)
    maxper = sims.max(axis=1)       # best match per HARD problem
    argmax = sims.argmax(axis=1)
    for i, row in enumerate(pairs):
        row["best_match"][name] = {"similarity": round(float(maxper[i]), 4),
                                   "reference_index": int(argmax[i]),
                                   "reference_text": texts[int(argmax[i])]}
    near = int((maxper >= 0.85).sum())
    lit = int((maxper >= 0.95).sum())
    union_near.update(np.where(maxper >= 0.85)[0].tolist())
    union_lit.update(np.where(maxper >= 0.95)[0].tolist())
    report["per_dataset"][name] = {
        "n_reference": len(texts),
        "near_0.85": near, "near_pct": round(near / len(hard_texts) * 100, 2),
        "literal_0.95": lit, "literal_pct": round(lit / len(hard_texts) * 100, 2),
        "max_similarity": round(float(maxper.max()), 4),
    }
    print(f"{name:22s} Nref={len(texts):5d}  near>=0.85: {near} ({near/len(hard_texts)*100:.2f}%)  "
          f"literal>=0.95: {lit}  max_sim: {maxper.max():.4f}")

report["union"] = {"near_0.85": len(union_near), "literal_0.95": len(union_lit)}
print(f"\nUNION any reference: near>=0.85 {len(union_near)}, literal>=0.95 {len(union_lit)}")

out = ROOT / "results" / "contamination_v4_report.json"
json.dump(report, open(out, "w"), indent=2)
json.dump(pairs, open(ROOT / "results" / "contamination_pairs.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"Saved: {out} and contamination_pairs.json")
