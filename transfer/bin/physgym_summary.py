#!/usr/bin/env python3
"""physgym_summary.py --root <OUT> [--step0 ARM --step10 ARM] — summarise a physgym-eval.sh campaign directory.

Layout: <OUT>/runs/<arm>/rep<k>/histories/baseline/<mode>/vllm_<arm>-rep<k>/experiment_<id>_{results.json, *_log.jsonl}
Success = results.json["is_correct"] (PhysGym's own definition: a `correct_hypothesis_found` event). Denominator is FIXED
at the number of problems in full_samples.json (97): a missing/crashed problem counts as a failure (the paper's
postprocess.py would silently drop it). Per (arm, mode): success % per rep, mean ± std over reps, SEM. Paired step10 − step0
over problems (per-problem success averaged over reps) with a paired bootstrap 95% CI and win/loss/tie counts.
Diagnostics per (arm, mode) from the event logs — the confounds the pre-registered decision rule needs:
  calls, median/p90 completion tokens, finish_reason=length rate, parse-failure rate (researcher_analysis with no experiments
  and no hypothesis but a non-empty raw_response), "Missing required parameters" hypothesis errors, judge routing
  (method llm / llm_failed / symbolic) and successes rescued by the judge.
Writes PHYSGYM_SUMMARY.json and PHYSGYM_SUMMARY.md into <OUT>.
"""
import argparse, glob, json, math, os, random, re, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--step0", default="step00-base")
ap.add_argument("--step10", default="step10-rl")
ap.add_argument("--n-problems", type=int, default=97)
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
N = a.n_problems

def q(xs, p):
    if not xs: return float("nan")
    xs = sorted(xs); k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1))))); return xs[k]
def sem(xs): return (st.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else float("nan")
def f(x, n=1): return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"

# res[arm][mode][rep][env_id] = is_correct ; diag[arm][mode] counters
res = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
diag = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
tok = defaultdict(lambda: defaultdict(list))
beh = defaultdict(lambda: defaultdict(lambda: {"iters": [], "experiments": [], "tests": [], "uniq_hyp": []}))
for hdir in glob.glob(os.path.join(a.root, "runs", "*", "rep*", "histories", "baseline", "*", "vllm_*")):
    parts = hdir.split(os.sep); arm = parts[-6]; rep = int(parts[-5].replace("rep", "")); mode = parts[-2]   # .../runs/<arm>/rep<k>/histories/baseline/<mode>/vllm_<tag>
    for rf in glob.glob(os.path.join(hdir, "experiment_*_results.json")):
        env_id = int(re.search(r"experiment_(\d+)_results\.json", os.path.basename(rf)).group(1))
        try: d = json.load(open(rf))
        except Exception as e: print(f"WARN unreadable {rf}: {e}"); continue
        res[arm][mode][rep][env_id] = bool(d.get("is_correct"))
        B = beh[arm][mode]
        B["iters"].append(d.get("iterations_completed") or 0)
        B["experiments"].append(100 - (d.get("final_sample_quota_remaining") if d.get("final_sample_quota_remaining") is not None else 100))
        B["tests"].append(2 - (d.get("final_test_quota_remaining") if d.get("final_test_quota_remaining") is not None else 2))
    D = diag[arm][mode]
    for lf in glob.glob(os.path.join(hdir, "experiment_*_log.jsonl")):
        D["logs"] += 1
        rescued_this = False; hyps = set()
        for line in open(lf, errors="replace"):
            try: ev = json.loads(line)
            except Exception: continue
            t = ev.get("event_type"); data = ev.get("data") or {}
            if t == "researcher_analysis":
                D["calls"] += 1
                if not data.get("proposed_experiments_count") and not str(data.get("current_hypothesis") or "").strip() and str(data.get("raw_response") or "").strip():
                    D["parse_fail"] += 1
                h = str(data.get("current_hypothesis") or "").strip()
                if h and h != "None": hyps.add(h)
                api = ((data.get("model_thinking") or {}).get("api_response") or {})
                ch = (api.get("choices") or [{}])[0]
                if ch.get("finish_reason") == "length": D["finish_length"] += 1
                ct = (api.get("usage") or {}).get("completion_tokens")
                if isinstance(ct, (int, float)): tok[arm][mode].append(ct)
            elif t == "hypothesis_testing":
                D["tests"] += 1
                ev_m = (((data.get("evaluation") or {}).get("equivalence_metrics") or {}).get("method"))
                if ev_m in ("llm", "llm_failed", "symbolic"): D[f"judge_{ev_m}"] += 1
                if ev_m == "llm" and (data.get("evaluation") or {}).get("is_correct"): rescued_this = True
                err = json.dumps(data)
                if "Missing required parameters" in err: D["missing_params"] += 1
            elif t == "correct_hypothesis_found":
                D["correct_events"] += 1
        if rescued_this: D["judge_rescued_success"] += 1
        beh[arm][mode]["uniq_hyp"].append(len(hyps))

rng = random.Random(a.seed)
out = {"root": a.root, "n_problems": N, "arms": {}, "paired": {}, "diagnostics": {}}
L = [f"# PhysGym summary — {a.root}", f"denominator fixed at {N} problems; missing results count as failures", "",
     "| arm | mode | reps | results found per rep | success % per rep | mean ± std | SEM |", "|---|---|---|---|---|---|---|"]
for arm in sorted(res):
    for mode in sorted(res[arm]):
        reps = sorted(res[arm][mode]); rates = []; found = []
        for k in reps:
            probs = res[arm][mode][k]; found.append(len(probs)); rates.append(100.0 * sum(probs.values()) / N)
        mean = st.mean(rates); std = st.stdev(rates) if len(rates) > 1 else float("nan")
        out["arms"][f"{arm}/{mode}"] = {"reps": reps, "results_found": found, "success_pct_per_rep": rates, "mean": mean, "std": std, "sem": sem(rates)}
        L.append(f"| {arm} | {mode} | {len(reps)} | {'/'.join(map(str, found))} | {', '.join(f(r) for r in rates)} | {f(mean)} ± {f(std)} | {f(sem(rates))} |")

L += ["", f"## Paired {a.step10} − {a.step0} (per-problem success averaged over reps; missing = failure)", "",
      "| mode | reps (step0/step10) | step0 % | step10 % | diff (pts) | paired bootstrap 95% CI | wins/losses/ties |", "|---|---|---|---|---|---|---|"]
for mode in sorted(set(res[a.step0]) & set(res[a.step10])):
    def per_problem(arm):
        reps = res[arm][mode]; ids = set(); [ids.update(p.keys()) for p in reps.values()]
        return {e: st.mean([1.0 if reps[k].get(e, False) else 0.0 for k in reps]) for e in ids}, len(reps)
    p0, k0 = per_problem(a.step0); p1, k1 = per_problem(a.step10)
    ids = sorted(set(p0) | set(p1))
    if not ids: continue
    d = [p1.get(e, 0.0) - p0.get(e, 0.0) for e in ids]
    # problems never attempted by either arm are absent from ids; they contribute 0 diff, so pad to N for the mean over all problems
    d_all = d + [0.0] * max(0, N - len(d))
    diff = 100.0 * st.mean(d_all)
    boots = sorted(100.0 * st.mean([d_all[rng.randrange(len(d_all))] for _ in range(len(d_all))]) for _ in range(a.boot))
    lo, hi = boots[int(0.025 * a.boot)], boots[int(0.975 * a.boot) - 1]
    wins = sum(1 for x in d if x > 0); losses = sum(1 for x in d if x < 0); ties = N - wins - losses
    s0 = 100.0 * sum(p0.values()) / N; s1 = 100.0 * sum(p1.values()) / N
    out["paired"][mode] = {"reps": [k0, k1], "step0_pct": s0, "step10_pct": s1, "diff_pts": diff, "ci95": [lo, hi], "wins": wins, "losses": losses, "ties": ties}
    L.append(f"| {mode} | {k0}/{k1} | {f(s0)} | {f(s1)} | {diff:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {wins}/{losses}/{ties} |")

L += ["", "## Diagnostics (all reps pooled)", "",
      "| arm | mode | logs | calls | median / p90 completion tokens | finish=length % | parse-fail % | missing-params | tests | judge llm / llm_failed / symbolic | successes rescued by judge |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for arm in sorted(diag):
    for mode in sorted(diag[arm]):
        D = diag[arm][mode]; T = tok[arm][mode]; calls = D["calls"] or 1
        row = {k: D[k] for k in D}; row.update({"tok_median": q(T, 0.5), "tok_p90": q(T, 0.9), "finish_length_pct": 100.0 * D["finish_length"] / calls, "parse_fail_pct": 100.0 * D["parse_fail"] / calls})
        out["diagnostics"][f"{arm}/{mode}"] = row
        L.append(f"| {arm} | {mode} | {D['logs']} | {D['calls']} | {f(q(T,0.5),0)} / {f(q(T,0.9),0)} | {f(row['finish_length_pct'])} | {f(row['parse_fail_pct'])} | {D['missing_params']} | {D['tests']} | {D['judge_llm']} / {D['judge_llm_failed']} / {D['judge_symbolic']} | {D['judge_rescued_success']} |")

L += ["", "## Behaviour (all problems, all reps, unconditional on success; mean ± std over problems)", "",
      "| arm | mode | iterations | experiments used (of 100) | oracle tests used (of 2) | unique hypotheses per problem |", "|---|---|---|---|---|---|"]
def ms(xs): return f"{st.mean(xs):.1f} ± {st.stdev(xs):.1f}" if len(xs) > 1 else "-"
for arm in sorted(beh):
    for mode in sorted(beh[arm]):
        B = beh[arm][mode]
        out.setdefault("behaviour", {})[f"{arm}/{mode}"] = {k: {"mean": st.mean(v), "std": st.stdev(v)} for k, v in B.items() if len(v) > 1}
        L.append(f"| {arm} | {mode} | {ms(B['iters'])} | {ms(B['experiments'])} | {ms(B['tests'])} | {ms(B['uniq_hyp'])} |")
json.dump(out, open(os.path.join(a.root, "PHYSGYM_SUMMARY.json"), "w"), indent=1)
open(os.path.join(a.root, "PHYSGYM_SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
