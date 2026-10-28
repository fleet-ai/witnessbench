#!/usr/bin/env python3
"""
summarize_wilt.py --root ROOT --repo REPO [--think off,on] [--base NAME] [--rl NAME,NAME] [--split full]
                  [--boot 10000] [--seed 0] [--k 5] [--sesoi-elim 0.05] [--sesoi-acc 0.08] [--gate-pp 5]

Summarise a launch_wilt.sh campaign: ROOT/think<on|off>/<arm>/rep<k>/results/sglang__<arm>_full_context.json
(+ calls.ndjson per rep). Each entry = one hidden rule (test_index 1..50) with the full conversation.

Per episode (recomputed from the conversation, not from the harness' aggregate fields):
  correct            harness points >= 1000 (final lambda equivalent to the hidden rule on the paper's input grid)
  outcome            correct | wrong_rule | invalid_lambda | aborted (no final guess) | no_reply (>=2 empty replies,
                     from the results field no_reply_events written by the patched harness)
  n_tests            "Result for input (...)" lines = distinct harness-evaluated test cases (incl. repeats)
  repeats            test triples identical to an earlier one (upstream bug fixed in main.py; recomputed here too)
  elim@k             PRIMARY (k = --k, default 5): fraction of a FIXED hypothesis set eliminated by the episode's FIRST k
                     tests (episodes with fewer than k tests use their final value; 0 tests -> 0). Set = the 50
                     full-split rules + 150 procedurally generated lambdas (harness.hypothesis_space, random.seed(0)),
                     replayed with the upstream HypothesisSpaceTracker. Count-adjusted: it compares the informativeness
                     of the tests at a fixed budget, so an arm that commits early is not penalised and an arm that never
                     commits is not rewarded.
  elim               SECONDARY: same fraction after ALL of the episode's tests. NOTE it is dominated by n_tests (a doom
                     loop of 30 tests scores higher than 5 sharp tests); read it together with n_tests.
  elim_per_test      SECONDARY: elim / n_tests (mean marginal elimination per test; undefined when n_tests = 0).
  complaints         harness format complaints ("Please make sure to send your response in the format", cut-off,
                     wrong arity, unparseable) = format-failure events
Per arm: per-rep means -> mean ± std (SEM). Paired RL − base over the 50 fixed rules (item = rule, value = mean over
reps), two-level bootstrap (rules, then reps within rule x arm) 95% CI, wins/losses/ties, pre-registered decision
(CI_lo>0 -> "RL higher", CI_hi<0 -> "RL lower", each tagged "(meaningful)" if |Δ|>=s else "(below SESOI)"; CI within
±s -> "null"; else "inconclusive") plus a secondary label (supported: CI_lo>0 & Δ>=s;
RL harmful: CI_hi<0; no transfer: CI_hi<s; else inconclusive).
Gates (per arm): complaint rate per model message, finish_reason=length % (calls.ndjson), aborted %, invalid-lambda %,
no-reply % (episodes with >=1 empty reply); |RL − base| > --gate-pp -> "confounded".
Infra: reps carrying an INFRA_TAINTED marker or any episode with infra_errors>0 (endpoint failure after retries) are
EXCLUDED and listed; they are re-run by launch_wilt.sh, never scored. Writes ROOT/WILT_SUMMARY.{md,json}.
"""
import argparse, json, math, os, random, re, statistics as st, sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--repo", required=True, help="patched riotgames/wilt checkout (for harness.test_cases / hypothesis_space)")
ap.add_argument("--think", default="off,on")
ap.add_argument("--base", default="qwen3.8-27b-base")
ap.add_argument("--rl", default="witnessgym-qwen3.8-27b-rl-broadcast-step20")
ap.add_argument("--split", default="full")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--hyp-seed", type=int, default=0, help="seed for the 150 generated hypotheses of the fixed set")
ap.add_argument("--k", type=int, default=5, help="test budget for the primary elim@k")
ap.add_argument("--sesoi-elim", type=float, default=0.05)
ap.add_argument("--sesoi-acc", type=float, default=0.08)
ap.add_argument("--gate-pp", type=float, default=5.0)
a = ap.parse_args()

sys.path.insert(0, a.repo)
from harness.test_cases import TESTS_FULL, TESTS_LITE  # noqa: E402
from harness.hypothesis_space import HypothesisSpaceTracker, generate_hypothesis_space, combine_hypotheses  # noqa: E402

TESTS = TESTS_FULL if a.split == "full" else TESTS_LITE
random.seed(a.hyp_seed)
FIXED_HYPS = combine_hypotheses(TESTS_FULL, generate_hypothesis_space(150))   # deterministic given --hyp-seed
N_HYPS = len(FIXED_HYPS)

RESULT_RE = re.compile(r"^Result for input \((.*)\): (True|False)\. Attempts Remaining: (\d+)$")
COMPLAINTS = ("Please make sure to send your response in the format", "Looks like your response may have been cut off",
              "Please provide exactly", "Looks like you didn't provide a test case", "Unable to parse test case",
              "Please wrap your function in backticks", "Unable to parse the lambda function")

def f(x, n=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
def mean(xs): xs = [x for x in xs if x is not None]; return st.mean(xs) if xs else None
def sd(xs): xs = [x for x in xs if x is not None]; return st.stdev(xs) if len(xs) > 1 else None
def sem(xs): xs = [x for x in xs if x is not None]; return st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None

def episode(entry):
    conv = entry.get("conversation_history") or []
    users = [m.get("content") or "" for m in conv if m.get("role") == "user"]
    assts = [m.get("content") or "" for m in conv if m.get("role") == "assistant"]
    tests = []
    for u in users:
        m = RESULT_RE.match(u.strip())
        if m:
            try: tup = tuple(float(x) for x in m.group(1).split(","))
            except ValueError: continue
            tests.append((tup, m.group(2) == "True"))
    seen = set(); repeats = 0
    for tup, _ in tests:
        if tup in seen: repeats += 1
        seen.add(tup)
    tracker = HypothesisSpaceTracker(FIXED_HYPS)
    curve = []   # elim after test 1, 2, ... (replay of the fixed set)
    for tup, res in tests:
        if len(tup) == 3:
            try: tracker.update(*tup, res)
            except Exception: pass
        curve.append((N_HYPS - len(tracker.remaining_hypotheses)) / N_HYPS)
    elim = curve[-1] if curve else 0.0
    elim_k = curve[min(a.k, len(curve)) - 1] if curve else 0.0
    elim_per_test = (elim / len(tests)) if tests else None
    last = users[-1] if users else ""
    # same predicate as the harness (_is_final_guess): "final guess:" or "### final guess"; a bare "final guess" in
    # prose ("I will not make my final guess yet") does not terminate an episode and must not count as a commitment
    committed = any(("final guess:" in s.lower() or "### final guess" in s.lower()) for s in assts)
    points = entry.get("points", 0) or 0
    no_reply_events = int(entry.get("no_reply_events") or 0)
    infra_errors = int(entry.get("infra_errors") or 0)
    if points >= 1000: outcome = "correct"
    elif last.startswith("Sorry, that's not the correct rule"): outcome = "wrong_rule"
    elif last.startswith("Invalid lambda function"): outcome = "invalid_lambda"
    elif no_reply_events >= 2 or ("no_reply_events" not in entry and (not assts or all(not s.strip() for s in assts[-2:]) and len(assts) <= 2)): outcome = "no_reply"
    elif committed and last.startswith(("Aborting", "Unable to parse the lambda")): outcome = "invalid_lambda"
    else: outcome = "aborted"
    complaints = sum(1 for u in users if u.startswith(COMPLAINTS))
    return {"idx": int(entry.get("test_index")), "correct": int(points >= 1000), "points": points, "outcome": outcome,
            "n_tests": len(tests), "repeats": repeats, "repeats_harness": entry.get("repeats"), "guesses_harness": entry.get("guesses"),
            "elim": elim, "elim_k": elim_k, "elim_per_test": elim_per_test, "elim_curve": curve,
            "n_asst": sum(1 for s in assts if s.strip()), "complaints": complaints, "committed": int(committed),
            "no_reply_events": no_reply_events, "infra_errors": infra_errors}

rng = random.Random(a.seed)
RL = [x for x in a.rl.split(",") if x]
out = {"root": a.root, "repo": a.repo, "params": vars(a), "n_fixed_hypotheses": N_HYPS, "conditions": {}}
L = [f"# WILT summary — {a.root}", "", f"Primary metric elim@{a.k} (fraction of {N_HYPS} fixed hypotheses eliminated by the first {a.k} tests); "
     f"secondary elim (all tests; dominated by n_tests), elim/test, accuracy, n_tests, repeats. SESOI elim@{a.k} and elim {a.sesoi_elim}, accuracy {a.sesoi_acc}; "
     f"paired two-level bootstrap {a.boot}x seed {a.seed}; gate {a.gate_pp} pp.", ""]
qual = "" if a.split == "full" else f"_{a.split}"
for think in [t for t in a.think.split(",") if t]:
    croot = Path(a.root) / f"think{think}"
    if not croot.exists(): continue
    eps = defaultdict(lambda: defaultdict(dict)); calls = defaultdict(lambda: defaultdict(list)); short = []; tainted = []
    for rdir in sorted(croot.glob("*/rep*/results")):
        arm = rdir.parts[-3]; rep = int(rdir.parts[-2].replace("rep", ""))
        res = rdir / f"sglang__{arm}{qual}_full_context.json"   # exact upstream name: results/<sanitized model>[_split]_full_context.json
        if not res.exists(): continue
        try: data = json.load(open(res))
        except Exception: continue
        E = {}
        for entry in data:
            if isinstance(entry, dict) and "test_index" in entry:
                e = episode(entry); E[e["idx"]] = e
        n_infra = sum(1 for e in E.values() if e["infra_errors"] > 0)
        if (rdir.parent / "INFRA_TAINTED").exists() or n_infra:
            tainted.append((think, arm, rep, n_infra)); continue   # endpoint failures: excluded, launcher re-runs the rep
        eps[arm][rep] = E
        if len(eps[arm][rep]) != len(TESTS): short.append((think, arm, rep, len(eps[arm][rep])))
        cl = res.parent.parent / "calls.ndjson"
        if cl.exists():
            for line in open(cl, errors="replace"):
                try: calls[arm][rep].append(json.loads(line))
                except Exception: pass
    if not eps: continue
    items = sorted({i for arm in eps for rep in eps[arm] for i in eps[arm][rep]})
    C = {"items": items, "arms": {}, "short_reps": short, "infra_tainted_reps": tainted}
    L += [f"## thinking = {think.upper()} — {len(items)} rules; reps with fewer than {len(TESTS)} episodes: {len(short)}; "
          f"infra-tainted reps (excluded, to be re-run): {len(tainted)}", "",
          f"| arm | reps | episodes/rep | accuracy mean ± std (SEM) | elim@{a.k} mean ± std (SEM) | elim (all tests) mean ± std (SEM) | elim/test | tests/episode | repeats/episode | committed % | outcome mix (pooled) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(eps):
        reps = sorted(eps[arm]); per = defaultdict(list)
        for k in reps:
            E = list(eps[arm][k].values())
            per["n"].append(len(E)); per["acc"].append(mean([e["correct"] for e in E])); per["elim"].append(mean([e["elim"] for e in E]))
            per["elim_k"].append(mean([e["elim_k"] for e in E])); per["ept"].append(mean([e["elim_per_test"] for e in E]))
            per["tests"].append(mean([e["n_tests"] for e in E])); per["rep"].append(mean([e["repeats"] for e in E])); per["commit"].append(mean([e["committed"] for e in E]))
        allE = [e for k in reps for e in eps[arm][k].values()]; mix = Counter(e["outcome"] for e in allE)
        allc = [c for k in reps for c in calls[arm][k] if "error" not in c]
        n_err = sum(1 for k in reps for c in calls[arm][k] if "error" in c)
        NA = sum(e["n_asst"] for e in allE) or 1
        gates = {"complaint_pct": 100 * sum(e["complaints"] for e in allE) / NA,
                 "trunc_pct": (100 * sum(1 for c in allc if c.get("finish_reason") == "length") / len(allc)) if allc else None,
                 "aborted_pct": 100 * mix.get("aborted", 0) / len(allE), "invalid_lambda_pct": 100 * mix.get("invalid_lambda", 0) / len(allE),
                 "no_reply_pct": 100 * sum(1 for e in allE if e["no_reply_events"] > 0) / len(allE),
                 "completion_tok_per_call": (sum(c.get("completion_tokens") or 0 for c in allc) / len(allc)) if allc else None,
                 "reasoning_tok_per_call": (sum(c.get("reasoning_tokens") or 0 for c in allc) / len(allc)) if allc else None,
                 "n_calls": len(allc), "endpoint_error_calls": n_err}
        C["arms"][arm] = {"reps": reps, "episodes_per_rep": per["n"], "acc_per_rep": per["acc"], "elim_per_rep": per["elim"], "elim_k_per_rep": per["elim_k"],
                          "acc_mean": mean(per["acc"]), "acc_std": sd(per["acc"]), "acc_sem": sem(per["acc"]),
                          "elim_k_mean": mean(per["elim_k"]), "elim_k_std": sd(per["elim_k"]), "elim_k_sem": sem(per["elim_k"]),
                          "elim_mean": mean(per["elim"]), "elim_std": sd(per["elim"]), "elim_sem": sem(per["elim"]), "elim_per_test_mean": mean(per["ept"]),
                          "tests_mean": mean(per["tests"]), "repeats_mean": mean(per["rep"]), "committed_mean": mean(per["commit"]), "outcome_mix": dict(mix), "gates": gates}
        L.append(f"| {arm} | {len(reps)} | {'/'.join(map(str, per['n']))} | {f(mean(per['acc']))} ± {f(sd(per['acc']))} ({f(sem(per['acc']))}) | "
                 f"{f(mean(per['elim_k']))} ± {f(sd(per['elim_k']))} ({f(sem(per['elim_k']))}) | "
                 f"{f(mean(per['elim']))} ± {f(sd(per['elim']))} ({f(sem(per['elim']))}) | {f(mean(per['ept']))} | {f(mean(per['tests']), 1)} | {f(mean(per['rep']), 2)} | {f(100 * (mean(per['commit']) or 0), 1)} | {dict(mix)} |")
    L += ["", "| arm | complaint % (per model msg) | truncation % (finish=length) | aborted % | invalid lambda % | no-reply % (>=1 empty reply) | completion tok/call | reasoning tok/call | calls | endpoint-error calls (must be 0 in scored reps) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(eps):
        g = C["arms"][arm]["gates"]
        L.append(f"| {arm} | {f(g['complaint_pct'], 1)} | {f(g['trunc_pct'], 1)} | {f(g['aborted_pct'], 1)} | {f(g['invalid_lambda_pct'], 1)} | {f(g['no_reply_pct'], 1)} | {f(g['completion_tok_per_call'], 0)} | {f(g['reasoning_tok_per_call'], 0)} | {g['n_calls']} | {g['endpoint_error_calls']} |")
    def per_item(arm, fn):
        return {it: [fn(eps[arm][k][it]) for k in eps[arm] if it in eps[arm][k]] for it in items}
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
            for mname, fn, s in ((f"elim@{a.k}", lambda e: e["elim_k"], a.sesoi_elim), ("elim", lambda e: e["elim"], a.sesoi_elim),
                                 ("accuracy", lambda e: float(e["correct"]), a.sesoi_acc), ("n_tests", lambda e: float(e["n_tests"]), None),
                                 ("repeats", lambda e: float(e["repeats"]), None), ("elim_per_test", lambda e: e["elim_per_test"], None)):
                B, R = per_item(a.base, fn), per_item(rl, fn)
                B = {it: [v for v in vs if v is not None] for it, vs in B.items()}; R = {it: [v for v in vs if v is not None] for it, vs in R.items()}
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
                              "decision": decide(D, lo, hi, s) if s is not None else ("CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0"),
                              "dig_label": dig_label(D, lo, hi, s) if s is not None else None,
                              "per_item": {str(it): diffs[it] for it in common}}
            gb, gr = C["arms"][a.base]["gates"], C["arms"][rl]["gates"]
            conf = [k for k in ("complaint_pct", "trunc_pct", "aborted_pct", "invalid_lambda_pct", "no_reply_pct")
                    if gb.get(k) is not None and gr.get(k) is not None and abs(gr[k] - gb[k]) > a.gate_pp]
            res["confound_gates_tripped"] = conf; C["paired"][rl] = res
            L += ["", f"### Paired `{rl}` − `{a.base}` (per-rule mean over reps; two-level bootstrap {a.boot}x)", "",
                  "| metric | rules | base | RL | Δ | 95% CI | wins/losses/ties | SESOI | decision (this package) | secondary label |", "|---|---|---|---|---|---|---|---|---|---|"]
            for m in (f"elim@{a.k}", "elim", "accuracy", "n_tests", "repeats", "elim_per_test"):
                if m in res:
                    r = res[m]
                    L.append(f"| {m}{' (PRIMARY)' if m == f'elim@{a.k}' else ''} | {r['n_items']} | {f(r['base'])} | {f(r['rl'])} | {r['delta']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['wins']}/{r['losses']}/{r['ties']} | {r['sesoi'] if r['sesoi'] is not None else '-'} | {r['decision']} | {r['dig_label'] or '-'} |")
            L.append(f"\nConfound gates tripped (|RL − base| > {a.gate_pp} pp): {conf if conf else 'none'}")
    if short:
        L += ["", "Reps with missing episodes (re-run with launch_wilt.sh play; skip logic re-runs them):"] + [f"- {t}/{arm}/rep{k}: {n} episodes" for t, arm, k, n in short]
    if tainted:
        L += ["", "Infra-tainted reps EXCLUDED (endpoint failure after retries; launch_wilt.sh re-runs them, they are never scored):"] + [f"- {t}/{arm}/rep{k}: {n} episode(s) with infra_errors>0" for t, arm, k, n in tainted]
    L.append(""); out["conditions"][think] = C

json.dump(out, open(os.path.join(a.root, "WILT_SUMMARY.json"), "w"), indent=1, default=str)
open(os.path.join(a.root, "WILT_SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
