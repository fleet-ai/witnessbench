#!/usr/bin/env python3
"""
summarize_f2f.py --root ROOT --repo REPO [--think on,off] [--base NAME] [--rl NAME,NAME] [--boot 10000] [--seed 0]
                 [--sesoi-if 0.05] [--sesoi-succ 0.06] [--gate-pp 5]

Summarise a launch_f2f.sh campaign (Failing-to-Falsify, Wason 2-4-6 numeric):
  ROOT/think<on|off>/rep<k>/ood_test/<arm>/baseline/<RULE>/inst<i>/{transcript,_judge_guesses,_judge_compatibility}.jsonl
Per-episode metrics come from the ORIGINAL analyzer (src.analyze_variant_metrics.analyze_instance, imported from REPO —
same code path as the paper's summary tables), plus transcript/call-log diagnostics computed here.

Primary metric  IF = incompatibility fraction per episode = incompatible / (incompatible + compatible) over the paper's
                window (solved: tests before the first judged-correct Announce; unsolved: all tests), OK-status compat
                rows only. Higher IF = more falsification attempts. The paper's pooled I:C_all (sum I / sum C) is also
                reported per arm x rep for comparability with Table 2.
Secondary       task success (judged-correct Announce within 45 pairs), first-guess-correct-before-any-test rate,
                tests before first correct (solved only), whitespace tokens per model turn.
Statistics      per arm: per-rep means -> mean ± std (SEM over reps). Paired RL − base over the 80 fixed items
                (rule, instance), item value = mean over reps; two-level bootstrap (items, then reps within item x arm)
                percentile 95% CI; wins/losses/ties. FMTABORT episodes (10 format retries exhausted, <91 model turns) are
                scored by the analyzer on the partial transcript (success only if a correct Announce preceded the abort;
                IF over the tests present) and counted in the invalid-episode gate. Episodes whose IF is undefined
                (I+C = 0: solved at the first Announce with zero tests, or no OK compat row) are excluded from the IF
                pairing but kept for success; their count is reported per arm ("IF undefined"). Pre-registered decision:
                CI_lo>0 -> "RL higher", CI_hi<0 -> "RL lower" (each "(meaningful)" if |Δ|>=s else "(below SESOI)");
                CI within [−s,+s] -> "null"; else "inconclusive". A secondary label (supported: CI_lo>0 & Δ>=s;
                RL harmful: CI_hi<0; no transfer: CI_hi<s; else inconclusive) is printed next to it.
Gates           per arm: format-failure turn rate (model turns that needed ≥1 retry), truncation rate (calls with
                finish_reason=length), stop-string cut rate (calls whose SGLang matched_stop is a string, i.e. the ```
                stop fired inside the reply/<meta>/thinking and cut it; finish_reason is "stop" for those), invalid-episode
                rate (transcript with <91 model turns), judge UNPARSEABLE rate, compat non-OK rate.
                |RL − base| > --gate-pp percentage points -> "confounded" flag on that comparison.
Writes ROOT/F2F_SUMMARY.md and ROOT/F2F_SUMMARY.json.
"""
import argparse, glob, json, math, os, random, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--repo", required=True, help="patched wason_experiments dir (for src.analyze_variant_metrics)")
ap.add_argument("--think", default="off,on")
ap.add_argument("--base", default="qwen3.8-27b-base")
ap.add_argument("--rl", default="witnessgym-qwen3.8-27b-rl-broadcast-step20")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sesoi-if", type=float, default=0.05)
ap.add_argument("--sesoi-succ", type=float, default=0.06)
ap.add_argument("--gate-pp", type=float, default=5.0)
a = ap.parse_args()

sys.path.insert(0, a.repo)
import src.analyze_variant_metrics as avm  # noqa: E402  (original analyzer; stdlib-only)

def f(x, n=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
def mean(xs): xs = [x for x in xs if x is not None]; return st.mean(xs) if xs else None
def sd(xs): xs = [x for x in xs if x is not None]; return st.stdev(xs) if len(xs) > 1 else None
def sem(xs): xs = [x for x in xs if x is not None]; return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None

def read_jsonl(p):
    out = []
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

def episode(inst_dir):
    """One (think, arm, rep, rule, inst) episode: analyzer metrics + our diagnostics."""
    transcript, guesses, compat = avm.find_three_files(inst_dir)
    metrics, issues = avm.analyze_instance(inst_dir)
    doc = read_jsonl(transcript) if transcript else []
    model_turns = [r for r in doc if r.get("role") == "model"]
    retry_recs = [r for r in doc if r.get("role") == "retry_trace"]
    n_retries = sum(max(0, len(r.get("retries") or []) - 1) for r in retry_recs)
    calls = read_jsonl(inst_dir / "calls.ndjson") if (inst_dir / "calls.ndjson").exists() else []
    ws_tokens = sum(len(re.findall(r"\S+", r.get("raw") or "")) for r in model_turns)
    e = {
        "rule": metrics["rule_id"], "inst": metrics["inst_name"], "has_transcript": transcript is not None,
        "judged": guesses is not None and compat is not None, "n_model_turns": len(model_turns), "complete": len(model_turns) >= 91,
        "success": int(metrics["task_complete"]), "tests_before_first": metrics["tests_before_first"],
        "first_guess_no_tests": int(metrics["first_guess_correct_no_tests"]),
        "I": int(metrics["incompat_ct_solved_pre"]) + int(metrics["incompat_ct_unsolved_all"]),
        "C": int(metrics["compat_ct_solved_pre"]) + int(metrics["compat_ct_unsolved_all"]),
        "IF": metrics["incompat_frac_solved_pre"] if metrics["task_complete"] else metrics["incompat_frac_unsolved_all"],
        "fmt_fail_turns": len(retry_recs), "retries": n_retries, "ws_tokens_per_turn": (ws_tokens / len(model_turns)) if model_turns else None,
        "n_calls": len(calls), "n_length": sum(1 for c in calls if c.get("finish_reason") == "length"),
        "n_stop_cut": sum(1 for c in calls if isinstance(c.get("matched_stop"), str)),
        "completion_tokens": sum((c.get("completion_tokens") or 0) for c in calls) or None,
        "reasoning_tokens": sum((c.get("reasoning_tokens") or 0) for c in calls) or None,
        "judge_unparseable": sum(1 for s in issues if s.startswith("JUDGE_UNPARSEABLE")),
        "compat_not_ok": sum(1 for s in issues if s.startswith("COMPAT_STATUS_NOT_OK")),
        "n_announce_judged": sum(1 for r in (read_jsonl(guesses) if guesses else []) if isinstance(r, dict) and "outcome" in r),
        "n_compat_rows": sum(1 for r in (read_jsonl(compat) if compat else []) if isinstance(r, dict) and "status" in r),
        "missing_files": [s for s in issues if s.startswith("MISSING_FILES")],
    }
    return e

rng = random.Random(a.seed)
RL = [x for x in a.rl.split(",") if x]
out = {"root": a.root, "repo": a.repo, "params": vars(a), "conditions": {}}
L = [f"# Failing-to-Falsify summary — {a.root}", "",
     f"Primary metric IF (incompatibility fraction, paper window); SESOI IF {a.sesoi_if}, success {a.sesoi_succ}; "
     f"paired two-level bootstrap {a.boot}x, seed {a.seed}; confound gate {a.gate_pp} pp.", ""]

for think in [t for t in a.think.split(",") if t]:
    croot = Path(a.root) / f"think{think}"
    if not croot.exists():
        continue
    # eps[arm][rep][(rule, inst)] = episode dict
    eps = defaultdict(lambda: defaultdict(dict)); unjudged = []; incomplete = []
    for inst_dir in sorted(croot.glob("rep*/ood_test/*/baseline/O*_*/inst*")):
        if not inst_dir.is_dir(): continue
        rep = int(inst_dir.parts[-6].replace("rep", "")); arm = inst_dir.parts[-4]
        e = episode(inst_dir)
        if not e["has_transcript"]: continue
        key = (e["rule"], e["inst"])
        if not e["complete"]: incomplete.append((think, arm, rep, key, e["n_model_turns"]))
        if not e["judged"]: unjudged.append((think, arm, rep, key)); continue   # cannot be scored yet
        eps[arm][rep][key] = e
    if not eps:
        continue
    items = sorted({k for arm in eps for rep in eps[arm] for k in eps[arm][rep]}, key=lambda k: (k[0], int(re.sub(r"\D", "", k[1]) or 0)))
    C = {"items": [f"{r}/{i}" for r, i in items], "arms": {}, "unjudged": len(unjudged), "incomplete": len(incomplete)}
    L += [f"## thinking = {think.upper()} — {len(items)} items; unjudged episodes (excluded): {len(unjudged)}; "
          f"incomplete transcripts (format-abort; analyzer-scored on the partial transcript, counted invalid): {len(incomplete)}", "",
          "| arm | reps | episodes/rep | success mean ± std (SEM) | IF mean ± std (SEM) | IF undefined (excluded) | pooled I:C_all per rep | first-guess-no-test | tests before 1st correct (solved) | ws tok/turn |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(eps):
        reps = sorted(eps[arm]); per = defaultdict(list)
        for k in reps:
            E = list(eps[arm][k].values())
            per["n"].append(len(E)); per["succ"].append(mean([e["success"] for e in E]))
            per["IF"].append(mean([e["IF"] for e in E if e["IF"] is not None]))
            I, Cc = sum(e["I"] for e in E), sum(e["C"] for e in E); per["ic_all"].append((I / Cc) if Cc else None)
            per["fg"].append(mean([e["first_guess_no_tests"] for e in E]))
            per["tb"].append(mean([e["tests_before_first"] for e in E if e["success"]]))
            per["tok"].append(mean([e["ws_tokens_per_turn"] for e in E]))
        allE = [e for k in reps for e in eps[arm][k].values()]
        n_if_undef = sum(1 for e in allE if e["IF"] is None)
        T = sum(e["n_model_turns"] for e in allE) or 1; NC = sum(e["n_calls"] for e in allE)
        gates = {"fmt_fail_turn_pct": 100 * sum(e["fmt_fail_turns"] for e in allE) / T,
                 "trunc_pct": (100 * sum(e["n_length"] for e in allE) / NC) if NC else None,
                 "stop_cut_pct": (100 * sum(e["n_stop_cut"] for e in allE) / NC) if NC else None,
                 "invalid_pct": 100 * sum(1 for e in allE if not e["complete"]) / len(allE),
                 "judge_unparseable_pct": 100 * sum(e["judge_unparseable"] for e in allE) / max(1, sum(e["n_announce_judged"] for e in allE)),
                 "compat_not_ok_pct": 100 * sum(e["compat_not_ok"] for e in allE) / max(1, sum(e["n_compat_rows"] for e in allE)),
                 "completion_tok_per_call": (sum(e["completion_tokens"] or 0 for e in allE) / NC) if NC else None,
                 "reasoning_tok_per_call": (sum(e["reasoning_tokens"] or 0 for e in allE) / NC) if NC else None}
        C["arms"][arm] = {"reps": reps, "episodes_per_rep": per["n"], "success_per_rep": per["succ"], "IF_per_rep": per["IF"],
                          "ic_all_per_rep": per["ic_all"], "first_guess_per_rep": per["fg"], "tests_before_per_rep": per["tb"],
                          "success_mean": mean(per["succ"]), "success_std": sd(per["succ"]), "success_sem": sem(per["succ"]),
                          "IF_mean": mean(per["IF"]), "IF_std": sd(per["IF"]), "IF_sem": sem(per["IF"]), "IF_undefined_episodes": n_if_undef, "gates": gates}
        L.append(f"| {arm} | {len(reps)} | {'/'.join(map(str, per['n']))} | {f(mean(per['succ']))} ± {f(sd(per['succ']))} ({f(sem(per['succ']))}) | "
                 f"{f(mean(per['IF']))} ± {f(sd(per['IF']))} ({f(sem(per['IF']))}) | {n_if_undef}/{len(allE)} | {', '.join(f(x, 2) for x in per['ic_all'])} | "
                 f"{f(mean(per['fg']))} | {f(mean(per['tb']), 1)} | {f(mean(per['tok']), 1)} |")
    L += ["", "| arm | format-fail turn % | truncation % (finish=length) | stop-string cut % (matched_stop is str) | invalid episode % | judge UNPARSEABLE % | compat non-OK % | completion tok/call | reasoning tok/call |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(eps):
        g = C["arms"][arm]["gates"]
        L.append(f"| {arm} | {f(g['fmt_fail_turn_pct'], 1)} | {f(g['trunc_pct'], 1)} | {f(g['stop_cut_pct'], 1)} | {f(g['invalid_pct'], 1)} | {f(g['judge_unparseable_pct'], 1)} | {f(g['compat_not_ok_pct'], 1)} | {f(g['completion_tok_per_call'], 0)} | {f(g['reasoning_tok_per_call'], 0)} |")

    # ---- paired RL − base
    def per_item(arm, fn):
        return {it: [fn(eps[arm][k][it]) for k in eps[arm] if it in eps[arm][k] and fn(eps[arm][k][it]) is not None] for it in items}
    def decide(D, lo, hi, s):   # this package's rule (direction from the CI, magnitude qualifier from the SESOI)
        if lo > 0: return "RL higher (meaningful)" if D >= s else "RL higher (below SESOI)"
        if hi < 0: return "RL lower (meaningful)" if D <= -s else "RL lower (below SESOI)"
        if lo >= -s and hi <= s: return "null (CI within SESOI)"
        return "inconclusive"
    def dig_label(D, lo, hi, s):   # secondary label rule, in order: supported / RL harmful / no transfer / inconclusive
        if lo > 0 and D >= s: return "supported"
        if hi < 0: return "RL harmful"
        if hi < s: return "no transfer"
        return "inconclusive"
    C["paired"] = {}
    if a.base in eps:
        for rl in RL:
            if rl not in eps: continue
            res = {}
            for mname, fn, s in (("IF", lambda e: e["IF"], a.sesoi_if), ("success", lambda e: float(e["success"]), a.sesoi_succ)):
                B, R = per_item(a.base, fn), per_item(rl, fn)
                common = [it for it in items if B[it] and R[it]]
                if not common: continue
                diffs = {it: st.mean(R[it]) - st.mean(B[it]) for it in common}
                D = st.mean(diffs.values()); boots = []
                for _ in range(a.boot):
                    its = [common[rng.randrange(len(common))] for _ in common]
                    boots.append(st.mean(st.mean([R[it][rng.randrange(len(R[it]))] for _ in R[it]]) - st.mean([B[it][rng.randrange(len(B[it]))] for _ in B[it]]) for it in its))
                boots.sort(); lo, hi = boots[int(0.025 * a.boot)], boots[int(0.975 * a.boot) - 1]
                w = sum(1 for it in common if diffs[it] > 0); l = sum(1 for it in common if diffs[it] < 0)
                res[mname] = {"n_items": len(common), "base": st.mean(st.mean(B[it]) for it in common), "rl": st.mean(st.mean(R[it]) for it in common),
                              "delta": D, "ci95": [lo, hi], "wins": w, "losses": l, "ties": len(common) - w - l, "sesoi": s,
                              "decision": decide(D, lo, hi, s), "dig_label": dig_label(D, lo, hi, s),
                              "per_item": {f"{r}/{i}": diffs[(r, i)] for (r, i) in common}}
            gb, gr = C["arms"][a.base]["gates"], C["arms"][rl]["gates"]
            conf = [k for k in ("fmt_fail_turn_pct", "trunc_pct", "stop_cut_pct", "invalid_pct", "judge_unparseable_pct", "compat_not_ok_pct")
                    if gb.get(k) is not None and gr.get(k) is not None and abs(gr[k] - gb[k]) > a.gate_pp]
            res["confound_gates_tripped"] = conf
            C["paired"][rl] = res
            L += ["", f"### Paired `{rl}` − `{a.base}` (per-item mean over reps; two-level bootstrap {a.boot}x)", "",
                  "| metric | items | base | RL | Δ | 95% CI | wins/losses/ties | SESOI | pre-registered decision (this package) | secondary label |", "|---|---|---|---|---|---|---|---|---|---|"]
            for m in ("IF", "success"):
                if m in res:
                    r = res[m]
                    L.append(f"| {m} | {r['n_items']} | {f(r['base'])} | {f(r['rl'])} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['wins']}/{r['losses']}/{r['ties']} | {r['sesoi']} | {r['decision']} | {r['dig_label']} |")
            L.append(f"IF-undefined episodes excluded from the IF pairing: base {C['arms'][a.base]['IF_undefined_episodes']}, RL {C['arms'][rl]['IF_undefined_episodes']} "
                     f"(items in IF row vs success row may therefore differ).")
            L.append(f"\nConfound gates tripped (|RL − base| > {a.gate_pp} pp): {conf if conf else 'none'}")
    if unjudged:
        L += ["", f"Unjudged episodes (run judge-guess/judge-compat, then re-summarize): {len(unjudged)}"] + [f"- {t}/{arm}/rep{k} {key[0]}/{key[1]}" for t, arm, k, key in unjudged[:40]]
    if incomplete:
        L += ["", "Incomplete transcripts (format abort after 10 retries; analyzer-scored on the partial transcript, counted invalid in the gate):"] + [f"- {t}/{arm}/rep{k} {key[0]}/{key[1]} model_turns={n}" for t, arm, k, key, n in incomplete]
    L.append("")
    out["conditions"][think] = C

json.dump(out, open(os.path.join(a.root, "F2F_SUMMARY.json"), "w"), indent=1, default=str)
open(os.path.join(a.root, "F2F_SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
