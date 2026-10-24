"""
Recompute ALL canonical numbers after removing the 2 confirmed near-duplicates
found by the contamination audit (17_contamination_v4.py):
  - PN_001083 : content-identical to AIME 1983 #5 (confounds the AIME baseline)
  - PN_000062 : content-identical to an OlympicArena/HMMT problem (shared source)
Both are in stage_b. Final released benchmark: N=881 (stage_a 200 + stage_b 681).

Reuses build_stage + metrics from 13_combined_883.py and 14_supplementary_v4.py.
Output: console + results/decontaminated_881_report.json
"""
from __future__ import annotations
import json, importlib.util
from pathlib import Path

S = Path(__file__).parent
def load_mod(fn, name):
    spec = importlib.util.spec_from_file_location(name, S / fn)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

c = load_mod("13_combined_883.py", "c13")
DROP = {"PN_001083", "PN_000062"}

pmm_a, ha = c.build_stage("stage_a")
pmm_b, hb = c.build_stage("stage_b")
hb = [p for p in hb if p not in DROP]   # decontaminate stage_b

pmm_c = {}
for (pid, m), v in pmm_a.items(): pmm_c[(f"A:{pid}", m)] = v
for (pid, m), v in pmm_b.items(): pmm_c[(f"B:{pid}", m)] = v
pids_c = [f"A:{p}" for p in ha] + [f"B:{p}" for p in hb]

out = {"dropped": sorted(DROP), "note": "decontaminated: 2 confirmed near-duplicates removed"}
print("=" * 70)
for label, pmm, pids in [("STAGE A", pmm_a, ha), ("STAGE B", pmm_b, hb), ("COMBINED", pmm_c, pids_c)]:
    ans, sol, gap, grand = c.grand_rel_gap(pmm, pids)
    rel, _ = c.per_model_first_rel(pmm, pids)
    lo, hi, _ = c.bootstrap_ci(pmm, pids)
    print(f"{label:9s} N={len(pids):4d}  ans={ans:.2f} sol={sol:.2f} abs=+{gap:.2f}  rel={rel:.2f}%  CI=[{lo:.2f},{hi:.2f}]")
    out[label.strip().lower().replace(' ', '_')] = {
        "n": len(pids), "ans": round(ans, 2), "sol": round(sol, 2), "abs": round(gap, 2),
        "rel": round(rel, 2), "ci": [round(lo, 2), round(hi, 2)]}

print("\nPER-MODEL (combined 881, strict+guard):")
out["per_model"] = []
for m, ans, sol, gap, rel, n in c.per_model_table(pmm_c, pids_c):
    print(f"  {m:24s} ans={ans:6.2f} sol={sol:6.2f} gap=+{gap:5.2f} rel={rel:5.2f}")
    out["per_model"].append({"model": m, "ans": round(ans, 2), "sol": round(sol, 2),
                             "gap": round(gap, 2), "rel": round(rel, 2)})

# threshold sweep + aggregator via module 14 helpers
s14 = load_mod("14_supplementary_v4.py", "c14")
pa, alla, easya, harda = s14.build_stage("stage_a")
pb, allb, easyb, hardb = s14.build_stage("stage_b")
hardb = [p for p in hardb if p not in DROP]
pmm14 = {}
for (pid, m), v in pa.items(): pmm14[(f"A:{pid}", m)] = v
for (pid, m), v in pb.items(): pmm14[(f"B:{pid}", m)] = v
hard14 = [f"A:{p}" for p in harda] + [f"B:{p}" for p in hardb]
easy14 = [f"A:{p}" for p in easya] + [f"B:{p}" for p in easyb]

print("\nTHRESHOLD sweep (combined 881, strict, per-model-first):")
out["threshold"] = {}
for thr in [5.5, 6.0, 6.5]:
    r, a, sol, g = s14.per_model_first_rel(pmm14, hard14, thr, True)
    out["threshold"][str(thr)] = round(r, 2)
    print(f"  mean>{thr}: rel={r:.2f}%")
lr, *_ = s14.per_model_first_rel(pmm14, hard14, 6.5, False)
sr, *_ = s14.per_model_first_rel(pmm14, hard14, 6.5, True)
out["aggregator"] = {"lenient": round(lr, 2), "strict": round(sr, 2)}
print(f"  aggregator lenient@6.5={lr:.2f}%  strict@6.5={sr:.2f}%")

er, ea, es, eg = s14.per_model_first_rel(pmm14, easy14, 6.5, True)
out["easy_pool"] = {"n": len(easy14), "rel": round(er, 2)}
print(f"\nEASY pool (unchanged): N={len(easy14)} rel={er:.2f}%")

save = S.parent / "results" / "decontaminated_881_report.json"
json.dump(out, open(save, "w"), indent=2)
print(f"\nSaved: {save}")
