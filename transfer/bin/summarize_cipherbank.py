#!/usr/bin/env python3
"""
summarize_cipherbank.py --root ROOT [--think on,off] [--mode nohint,hint] [--base NAME] [--rl NAME,NAME] [--boot 10000] [--seed 0]
                        [--sesoi-acc 0.03] [--sesoi-lev 0.02] [--gate-pp 5] [--hint-trigger 0.05] [--floor 0.01] [--expected-rows N]

Summarise a launch_cipherbank.sh campaign: ROOT/think<on|off>/<nohint|hint>/<served-name>/rep<k>/{results,calls}.ndjson
(one results line per finished item; the LAST line per item wins; a record with error != null is an endpoint failure after the
in-run re-run and is scored 0 — is_correct False, Levenshtein 0 — and counted in the endpoint-error gate).

Completeness  the EXPECTED grid (arms x reps x items) is derived from ROOT/provenance/play_think<T>_<mode>_*.json (config.arms,
             config.reps, config.rows; rows 'all' = 262 -> 2,358 items; --expected-rows overrides) unioned with what was observed.
             Every (arm, rep) with an item lacking ANY record is printed as a MISSING line; the condition is marked INCOMPLETE,
             every paired row gets decision "no decision (INCOMPLETE)", the hint trigger and predictions print "not evaluable".
             Pre-registration §5.6: no item is ever dropped — an item missing for one arm is a bug to fix and re-run, not data.
Metrics (scoring is the benchmark's own: utils.tools.compare_strings exact match after removing spaces, case-insensitive, and
utils.tools.levenshtein_similarity = 1 − Levenshtein(lower)/max_len; both computed by cb_driver.py at run time):
  PRIMARY      exact-match accuracy on the CUSTOM family (DualAvgCode, ParityShift, WordShift; 786 items) — uncontaminated by pretraining
  CO-PRIMARY   exact-match accuracy over all 2,358 items
  SECONDARY    Levenshtein similarity (custom, overall); per-algorithm, per-family, per-tier (paper T1–T5) tables
Statistics   per arm: per-rep means -> mean ± std (SEM over reps). Paired RL − base over items (item value = mean over reps),
             two-level bootstrap (resample items, then reps within item x arm; kmax = max reps over the two arms), --boot resamples,
             95% percentile CI, wins/losses/ties. A one-level (items only) CI is printed as a sensitivity column; decisions use the two-level CI.
Decision     (pre-registered, on the two-level CI): CI_hi < 0 -> "RL worse"; CI within [−SESOI, +SESOI] -> "null";
             CI_lo > +SESOI -> "RL better"; otherwise "inconclusive". SESOI: accuracy 3 pp, Levenshtein 0.02.
             Every paired row carries decision_ci (bare CI label), confounded, incomplete and decision (= decision_ci only when
             neither confounded nor incomplete; otherwise "descriptive/CONFOUNDED (ci: ...)" or "no decision (INCOMPLETE)").
Gates        per arm over all returned calls: truncation % (finish_reason == length), format-failure % (no <result> tag), empty-answer %;
             endpoint-error % over all items. |RL − base| > --gate-pp pp on any -> the comparison is "confounded/descriptive".
             Also: a condition whose complete records carry more than one max_tokens value, or whose arms ran under different
             max_tokens, trips the gate "max_tokens" (a resume under another budget; the driver re-runs such items — this catches leftovers).
Confirmatory ON is the pre-registered primary dialect. Per RL arm: ON passes all gates -> ON confirmatory; ON confounded and OFF
             passes all gates -> OFF confirmatory; both confounded -> "no decision"; anything incomplete/not run -> pending. Printed
             as CONFIRMATORY lines and stored under "confirmatory" in the JSON.
Floor        if base and RL both < --floor on a metric the row is annotated FLOOR (no dynamic range; decision still printed).
Hint trigger the pre-registered secondary hint-mode run is due iff base no-hint CUSTOM accuracy (primary dialect) < --hint-trigger.
Predictions  P1 no RL arm is "RL better" on the primary metric — evaluated ONLY on non-confounded, complete rows (a confounded arm
             prints "not evaluable"); P2 credit-C truncation % >= base (a statement about the gate quantity itself, no guard needed);
             P3 every arm scores higher on the pretraining-recognisable ciphers (Rot13/Atbash/Reverse) than on the custom family (within-arm).
Writes ROOT/CIPHERBANK_SUMMARY.md and ROOT/CIPHERBANK_SUMMARY.json.
"""
import argparse, glob, json, math, os, statistics as st
from collections import defaultdict
import numpy as np

CIPHERS = ["Rot13", "Atbash", "Polybius", "Vigenere", "Reverse", "SwapPairs", "ParityShift", "DualAvgCode", "WordShift"]
FAMILY = {"Rot13": "substitution", "Atbash": "substitution", "Polybius": "substitution", "Vigenere": "substitution",
          "Reverse": "transposition", "SwapPairs": "transposition", "ParityShift": "custom", "DualAvgCode": "custom", "WordShift": "custom"}
TIER = {"Rot13": "T1", "Reverse": "T1", "Atbash": "T2", "WordShift": "T2", "DualAvgCode": "T3", "Polybius": "T3", "ParityShift": "T4", "SwapPairs": "T4", "Vigenere": "T5"}
RECOG = {"Rot13", "Atbash", "Reverse"}
ARM_NAME = {"base": "qwen3.8-27b-base", "step20": "witnessgym-qwen3.8-27b-rl-broadcast-step20"}
N_ROWS_ALL = 262                                                          # pinned HF data (md5 592d88ac…): 262 plaintexts

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--think", default="on,off")
ap.add_argument("--mode", default="nohint,hint")
ap.add_argument("--base", default="qwen3.8-27b-base")
ap.add_argument("--rl", default="witnessgym-qwen3.8-27b-rl-broadcast-step20")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sesoi-acc", type=float, default=0.03)
ap.add_argument("--sesoi-lev", type=float, default=0.02)
ap.add_argument("--gate-pp", type=float, default=5.0)
ap.add_argument("--hint-trigger", type=float, default=0.05)
ap.add_argument("--floor", type=float, default=0.01)
ap.add_argument("--primary-think", default="on")
ap.add_argument("--expected-rows", type=int, default=0, help="rows intended per condition (default: from provenance config.rows; 'all' = 262)")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

def f(x, n=3): return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{n}f}"
def pct(x, n=1): return "-" if x is None else f"{100*x:.{n}f}"
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
def item_key(s): return (CIPHERS.index(s.split("/")[0]), s)

def load_condition(croot):
    """recs[arm][rep][item] = last results record; calls[arm] = list of call records"""
    recs = defaultdict(lambda: defaultdict(dict)); calls = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(croot, "*", "rep*", "results.ndjson"))):
        arm = p.split(os.sep)[-3]; rep = int(p.split(os.sep)[-2][3:])
        for r in read_jsonl(p): recs[arm][rep][r["item"]] = r
        cp = os.path.join(os.path.dirname(p), "calls.ndjson")
        if os.path.exists(cp): calls[arm] += read_jsonl(cp)
    return recs, calls

def expected_grid(think, mode, recs):
    """(arms, reps, items, source) intended for this condition: provenance play_*.json config unioned with what was observed."""
    provs = sorted(glob.glob(os.path.join(a.root, "provenance", f"play_think{think}_{mode}_*.json")))
    rows, arms, reps = [], set(), set()
    for p in provs:
        try: cfg = json.load(open(p)).get("config", {})
        except Exception: continue
        r = cfg.get("rows"); rows.append(N_ROWS_ALL if r == "all" else int(r))
        arms |= {ARM_NAME.get(x, x) for x in str(cfg.get("arms", "")).split(",") if x}
        reps |= {int(x) for x in str(cfg.get("reps", "")).split(",") if x}
    n_rows = a.expected_rows or (max(rows) if rows else N_ROWS_ALL)
    src = f"--expected-rows {a.expected_rows}" if a.expected_rows else (f"{len(provs)} provenance file(s)" if provs else "no provenance -> 262 rows assumed")
    items = {f"{c}/{r:03d}" for c in CIPHERS for r in range(n_rows)} | {it for arm in recs for rep in recs[arm] for it in recs[arm][rep]}
    arms |= set(recs); reps |= {rep for arm in recs for rep in recs[arm]}
    return sorted(arms, key=lambda x: (x != a.base, x)), sorted(reps), sorted(items, key=item_key), src

def val(r, m):   # scored value of a record for metric m (errored records -> 0, pre-registered)
    if r.get("error") is not None: return 0.0
    return float(bool(r["is_correct"])) if m == "acc" else float(r["levenshtein"])

def item_matrix(recs_arm, items, m, kmax):
    """returns V (n_items x kmax, NaN padded), k (n_items) from recs_arm[rep][item]"""
    reps = sorted(recs_arm)
    V = np.full((len(items), kmax), np.nan); k = np.zeros(len(items), dtype=int)
    for i, it in enumerate(items):
        vals = [val(recs_arm[rep][it], m) for rep in reps if it in recs_arm[rep]]
        V[i, :len(vals)] = vals; k[i] = len(vals)
    return V, k

def two_level_boot(VB, kB, VR, kR, boot, chunk=500):
    """paired Δ = mean_i( mean_reps R_i − mean_reps B_i ) with items resampled, then reps within (item, arm) resampled.
    kmax draws per (item, arm) with kmax = max reps over both arms; each draw is uniform over that item's k_i available reps."""
    n, kmax = VB.shape
    VB0 = np.nan_to_num(VB); VR0 = np.nan_to_num(VR)
    out = np.empty(boot)
    for s in range(0, boot, chunk):
        m = min(chunk, boot - s)
        idx = rng.integers(0, n, size=(m, n))
        kb = kB[idx]; kr = kR[idx]                                   # (m, n)
        u = rng.random((m, n, kmax)); rb = np.floor(u * kb[..., None]).astype(int)
        u = rng.random((m, n, kmax)); rr = np.floor(u * kr[..., None]).astype(int)
        vb = np.take_along_axis(VB0[idx], rb, axis=2).mean(axis=2)
        vr = np.take_along_axis(VR0[idx], rr, axis=2).mean(axis=2)
        out[s:s + m] = (vr - vb).mean(axis=1)
    out.sort()
    return out[int(0.025 * boot)], out[max(0, int(0.975 * boot) - 1)]

def one_level_boot(d, boot, chunk=500):
    """sensitivity only: resample items of the per-item paired difference (rep noise enters once, through the item means)."""
    n = len(d); out = np.empty(boot)
    for s in range(0, boot, chunk):
        m = min(chunk, boot - s); idx = rng.integers(0, n, size=(m, n)); out[s:s + m] = d[idx].mean(axis=1)
    out.sort()
    return out[int(0.025 * boot)], out[max(0, int(0.975 * boot) - 1)]

def decide(lo, hi, s):
    if hi < 0: return "RL worse"
    if lo >= -s and hi <= s: return "null"
    if lo > s: return "RL better"
    return "inconclusive"

def paired(recsB, recsR, items, m, sesoi, conf, incomplete):
    items = [it for it in items if any(it in recsB[r] for r in recsB) and any(it in recsR[r] for r in recsR)]
    if not items: return None
    kmax = max(len(recsB), len(recsR))
    VB, kB = item_matrix(recsB, items, m, kmax); VR, kR = item_matrix(recsR, items, m, kmax)
    mb = np.nanmean(VB, axis=1); mr = np.nanmean(VR, axis=1); d = mr - mb
    lo, hi = two_level_boot(VB, kB, VR, kR, a.boot); lo1, hi1 = one_level_boot(d, a.boot)
    dci = decide(lo, hi, sesoi)
    final = dci if not conf and not incomplete else ("no decision (INCOMPLETE)" if incomplete else f"descriptive/CONFOUNDED (ci: {dci})")
    return {"n_items": len(items), "base": float(mb.mean()), "rl": float(mr.mean()), "delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "ci95_1level_sensitivity": [float(lo1), float(hi1)], "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
            "sesoi": sesoi, "decision_ci": dci, "confounded": bool(conf), "confound_gates_tripped": list(conf), "incomplete": bool(incomplete), "decision": final,
            "floor": bool(mb.mean() < a.floor and mr.mean() < a.floor)}

RL = [x for x in a.rl.split(",") if x]
out = {"root": a.root, "params": vars(a), "conditions": {}}
L = [f"# CipherBank summary — {a.root}", "",
     f"Scoring = benchmark's own (exact match after space removal, case-insensitive; Levenshtein similarity). PRIMARY = exact-match accuracy on the "
     f"CUSTOM family (DualAvgCode/ParityShift/WordShift); CO-PRIMARY = overall. SESOI acc {a.sesoi_acc} / Levenshtein {a.sesoi_lev}; paired two-level "
     f"bootstrap {a.boot}x seed {a.seed} (one-level CI shown as sensitivity only); confound gate {a.gate_pp} pp (+ max_tokens mixing); endpoint failures scored 0; "
     f"missing records -> INCOMPLETE, no decision. Rot13/Atbash/Reverse are pretraining-recognisable: reported, not headlined. "
     f"'calls' = driver attempts (each <= 6 HTTP requests: SDK max_retries=2 x wrapper max_num_retries=2).", ""]
hint_trigger = {}
for think in [t for t in a.think.split(",") if t]:
    for mode in [m for m in a.mode.split(",") if m]:
        croot = os.path.join(a.root, f"think{think}", mode)
        if not os.path.isdir(croot): continue
        recs, calls = load_condition(croot)
        if not recs: continue
        arms, exp_reps, items, exp_src = expected_grid(think, mode, recs)
        groups = {"custom": [it for it in items if FAMILY[it.split("/")[0]] == "custom"], "overall": items,
                  "substitution": [it for it in items if FAMILY[it.split("/")[0]] == "substitution"], "transposition": [it for it in items if FAMILY[it.split("/")[0]] == "transposition"],
                  "recognisable": [it for it in items if it.split("/")[0] in RECOG]}
        for c in CIPHERS: groups[c] = [it for it in items if it.split("/")[0] == c]
        for t in ["T1", "T2", "T3", "T4", "T5"]: groups[t] = [it for it in items if TIER[it.split("/")[0]] == t]
        # ---- completeness: every expected (arm, rep) must have a record (any record; errored = scored 0) for every expected item
        missing_lines = []; missing = {}
        for arm in arms:
            for rep in exp_reps:
                have = set(recs[arm][rep]) if (arm in recs and rep in recs[arm]) else set()
                miss = [it for it in items if it not in have]
                missing[f"{arm}/rep{rep}"] = {"n_missing": len(miss), "n_expected": len(items), "n_found": len(have), "ids": miss[:50]}
                if miss:
                    missing_lines.append(f"MISSING think={think} mode={mode} arm={arm} rep={rep}: {len(miss)} of {len(items)} items have NO record"
                                         + (" (rep dir absent)" if not have else "") + f" — {', '.join(miss[:20])}{' …' if len(miss) > 20 else ''}")
        incomplete = bool(missing_lines)
        C = {"n_items_expected": len(items), "expected_source": exp_src, "expected_arms": arms, "expected_reps": exp_reps, "incomplete": incomplete,
             "missing": missing, "arms": {}, "paired": {}}
        L += [f"## thinking = {think.upper()}, mode = {mode} — {len(items)} items expected ({len(groups['custom'])} custom; source: {exp_src}); arms: {', '.join(arms)}; reps: {exp_reps}"
              + ("  **INCOMPLETE — missing records below; no decision until complete**" if incomplete else "  (complete)"), ""]
        L += missing_lines + ([""] if missing_lines else [])
        # ---- per-arm table
        L += ["| arm | reps | items found per rep | missing | acc custom (PRIMARY) mean ± std (SEM) | acc overall (CO-PRIMARY) | acc substitution | acc transposition | acc recognisable (not headline) | Lev custom | Lev overall | endpoint errors (scored 0) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in arms:
            if arm not in recs:
                C["arms"][arm] = {"reps": [], "items_per_rep": [], "per_rep": {}, "summary": {}, "gates": {}, "endpoint_errors": 0, "absent": True}
                L.append(f"| {arm} | 0 | - | {sum(missing[f'{arm}/rep{r}']['n_missing'] for r in exp_reps)} | (no results) |  |  |  |  |  |  |  |"); continue
            reps = sorted(recs[arm]); per = defaultdict(list)
            for rep in reps:
                R = recs[arm][rep]; per["n"].append(len(R))
                for g in ("custom", "overall", "substitution", "transposition", "recognisable"):
                    xs = [val(R[it], "acc") for it in groups[g] if it in R]; per[f"acc_{g}"].append(mean(xs))
                    xs = [val(R[it], "lev") for it in groups[g] if it in R]; per[f"lev_{g}"].append(mean(xs))
                for c in CIPHERS + ["T1", "T2", "T3", "T4", "T5"]:
                    xs = [val(R[it], "acc") for it in groups[c] if it in R]; per[f"acc_{c}"].append(mean(xs))
                    xs = [val(R[it], "lev") for it in groups[c] if it in R]; per[f"lev_{c}"].append(mean(xs))
            allR = [r for rep in reps for r in recs[arm][rep].values()]
            ok = [r for r in allR if r.get("error") is None]; n_ok = len(ok) or 1
            cl = calls[arm]
            gates = {"trunc_pct": 100 * sum(1 for r in ok if r.get("finish_reason") == "length") / n_ok,
                     "fmt_fail_pct": 100 * sum(1 for r in ok if not r.get("has_result_tag")) / n_ok,
                     "empty_pct": 100 * sum(1 for r in ok if r.get("empty_answer")) / n_ok,
                     "endpoint_err_pct": 100 * sum(1 for r in allR if r.get("error") is not None) / max(1, len(allR)),
                     "max_tokens_seen": sorted({r.get("max_tokens") for r in ok}, key=lambda x: (x is None, x)),
                     "n_calls": len(cl), "n_call_errors": sum(1 for c in cl if c.get("error") is not None),
                     "completion_tok_per_call": mean([r.get("completion_tokens") for r in ok]), "reasoning_tok_per_call": mean([r.get("reasoning_tokens") for r in ok]),
                     "reasoning_chars": mean([r.get("reasoning_chars") for r in ok]), "response_chars": mean([r.get("response_chars") for r in ok]),
                     "latency_s": mean([r.get("latency_s") for r in ok]), "acc_given_tag": mean([val(r, "acc") for r in ok if r.get("has_result_tag")])}
            C["arms"][arm] = {"reps": reps, "items_per_rep": per["n"], "n_missing": sum(missing[f"{arm}/rep{r}"]["n_missing"] for r in exp_reps),
                              "per_rep": {k: v for k, v in per.items() if k != "n"},
                              "summary": {k: {"mean": mean(v), "std": sd(v), "sem": sem(v)} for k, v in per.items() if k != "n"}, "gates": gates,
                              "endpoint_errors": sum(1 for r in allR if r.get("error") is not None)}
            s = C["arms"][arm]["summary"]
            cell = lambda k: f"{pct(s[k]['mean'])} ± {pct(s[k]['std'])} ({pct(s[k]['sem'])})"
            L.append(f"| {arm} | {len(reps)} | {'/'.join(map(str, per['n']))} of {len(items)} | {C['arms'][arm]['n_missing']} | {cell('acc_custom')} | {cell('acc_overall')} | {pct(s['acc_substitution']['mean'])} | "
                     f"{pct(s['acc_transposition']['mean'])} | {pct(s['acc_recognisable']['mean'])} | {f(s['lev_custom']['mean'])} ± {f(s['lev_custom']['std'])} | "
                     f"{f(s['lev_overall']['mean'])} ± {f(s['lev_overall']['std'])} | {C['arms'][arm]['endpoint_errors']}/{len(allR)} |")
        present = [arm for arm in arms if arm in recs]
        # ---- per-algorithm and per-tier tables (accuracy %, mean over reps)
        L += ["", "Per-algorithm exact-match accuracy % (mean over reps; family/tier in the header; * = pretraining-recognisable):", "",
              "| arm | " + " | ".join(f"{c}{'*' if c in RECOG else ''} ({FAMILY[c][:3]}/{TIER[c]})" for c in CIPHERS) + " |", "|---|" + "---|" * len(CIPHERS)]
        for arm in present:
            s = C["arms"][arm]["summary"]; L.append(f"| {arm} | " + " | ".join(pct(s[f'acc_{c}']['mean']) for c in CIPHERS) + " |")
        L += ["", "Per-tier exact-match accuracy % / Levenshtein (mean over reps): T1 Rot13+Reverse, T2 Atbash+WordShift, T3 DualAvgCode+Polybius, T4 ParityShift+SwapPairs, T5 Vigenere", "",
              "| arm | T1 | T2 | T3 | T4 | T5 |", "|---|---|---|---|---|---|"]
        for arm in present:
            s = C["arms"][arm]["summary"]; L.append(f"| {arm} | " + " | ".join(f"{pct(s[f'acc_{t}']['mean'])} / {f(s[f'lev_{t}']['mean'], 2)}" for t in ["T1", "T2", "T3", "T4", "T5"]) + " |")
        # ---- gates + behaviour
        L += ["", "| arm | truncation % (finish=length) | format-failure % (no <result>) | empty-answer % | endpoint-error % (items) | max_tokens seen | driver attempts (errors) | completion tok/call | reasoning tok/call | reasoning chars | response chars | latency s | acc given tag |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm in present:
            g = C["arms"][arm]["gates"]
            L.append(f"| {arm} | {f(g['trunc_pct'], 1)} | {f(g['fmt_fail_pct'], 1)} | {f(g['empty_pct'], 1)} | {f(g['endpoint_err_pct'], 1)} | {','.join(map(str, g['max_tokens_seen']))} | {g['n_calls']} ({g['n_call_errors']}) | "
                     f"{f(g['completion_tok_per_call'], 0)} | {f(g['reasoning_tok_per_call'], 0)} | {f(g['reasoning_chars'], 0)} | {f(g['response_chars'], 0)} | {f(g['latency_s'], 1)} | {pct(g['acc_given_tag'])} |")
        # ---- paired
        if a.base in recs:
            for rl in RL:
                if rl not in recs: continue
                res = {}
                gb, gr = C["arms"][a.base]["gates"], C["arms"][rl]["gates"]
                conf = [k for k in ("trunc_pct", "fmt_fail_pct", "empty_pct", "endpoint_err_pct") if abs(gr[k] - gb[k]) > a.gate_pp]
                if len(gb["max_tokens_seen"]) > 1 or len(gr["max_tokens_seen"]) > 1 or gb["max_tokens_seen"] != gr["max_tokens_seen"]:
                    conf.append("max_tokens")                             # mixed budgets within an arm or different budgets across arms
                specs = [("acc_custom (PRIMARY)", "custom", "acc", a.sesoi_acc), ("acc_overall (CO-PRIMARY)", "overall", "acc", a.sesoi_acc),
                         ("lev_custom", "custom", "lev", a.sesoi_lev), ("lev_overall", "overall", "lev", a.sesoi_lev),
                         ("acc_substitution", "substitution", "acc", a.sesoi_acc), ("acc_transposition", "transposition", "acc", a.sesoi_acc),
                         ("acc_recognisable*", "recognisable", "acc", a.sesoi_acc)]
                specs += [(f"acc_{c}{'*' if c in RECOG else ''}", c, "acc", a.sesoi_acc) for c in CIPHERS] + [(f"acc_{t}", t, "acc", a.sesoi_acc) for t in ["T1", "T2", "T3", "T4", "T5"]]
                for label, g, m, s in specs:
                    r = paired(recs[a.base], recs[rl], groups[g], m, s, conf, incomplete)
                    if r: res[label] = r
                res["confound_gates_tripped"] = conf; res["confounded"] = bool(conf); res["incomplete"] = incomplete
                C["paired"][rl] = res
                L += ["", f"### Paired `{rl}` − `{a.base}` (item value = mean over reps; two-level bootstrap {a.boot}x; 1-level CI = sensitivity only). Confound gates tripped: {conf if conf else 'none'}"
                      + (" -> ALL rows below are CONFOUNDED / descriptive only" if conf else "") + (" -> INCOMPLETE: no decision" if incomplete else ""), "",
                      "| metric | items | base | RL | Δ | 95% CI (2-level, decides) | 95% CI (1-level, sensitivity) | wins/losses/ties | SESOI | decision | note |", "|---|---|---|---|---|---|---|---|---|---|---|"]
                for label, _, m, _ in specs:
                    if label not in res: continue
                    r = res[label]; fm = (lambda x: pct(x, 2)) if m == "acc" else (lambda x: f(x, 4))
                    note = ("FLOOR " if r["floor"] else "") + ("INCOMPLETE " if incomplete else "") + ("CONFOUNDED " if conf else "") + ("descriptive" if label.split(" ")[0] not in ("acc_custom", "acc_overall", "lev_custom", "lev_overall") else "")
                    L.append(f"| {label} | {r['n_items']} | {fm(r['base'])} | {fm(r['rl'])} | {('+' if r['delta'] >= 0 else '')}{fm(r['delta'])} | [{fm(r['ci95'][0])}, {fm(r['ci95'][1])}] | "
                             f"[{fm(r['ci95_1level_sensitivity'][0])}, {fm(r['ci95_1level_sensitivity'][1])}] | {r['wins']}/{r['losses']}/{r['ties']} | {r['sesoi']} | {r['decision']} | {note.strip()} |")
        # ---- hint trigger (evaluated on the primary dialect, no-hint, base arm, custom family)
        if mode == "nohint" and a.base in recs:
            b = C["arms"][a.base]["summary"]["acc_custom"]["mean"]
            trig = (b is not None and b < a.hint_trigger) if not incomplete else None
            hint_trigger[think] = {"base_acc_custom_nohint": b, "threshold": a.hint_trigger, "triggered": trig, "primary_dialect": think == a.primary_think, "incomplete": incomplete}
            tag = f"HINT TRIGGER (think={think}{', PRIMARY dialect' if think == a.primary_think else ''}): "
            L += ["", tag + (f"not evaluable (INCOMPLETE; base no-hint custom accuracy so far = {pct(b)}%)" if incomplete else
                             f"base no-hint custom accuracy = {pct(b)}% {'<' if trig else '>='} {100*a.hint_trigger:.0f}% -> secondary hint-mode run {'TRIGGERED' if trig else 'NOT triggered'}")]
        # ---- directional predictions
        if a.base in recs and C["paired"]:
            P = {}
            if incomplete:
                P["P1 no RL arm 'RL better' on acc_custom"] = P["P2 credit-C truncation % >= base"] = P["P3 every arm: acc recognisable > acc custom"] = "not evaluable (INCOMPLETE)"
            else:
                ev = {rl: C["paired"][rl].get("acc_custom (PRIMARY)") for rl in C["paired"]}
                falsified = [rl for rl, r in ev.items() if r and not r["confounded"] and r["decision_ci"] == "RL better"]
                held = [rl for rl, r in ev.items() if r and not r["confounded"] and r["decision_ci"] != "RL better"]
                nev = [f"{rl} (confounded: {','.join(C['paired'][rl]['confound_gates_tripped'])})" for rl, r in ev.items() if r and r["confounded"]]
                p1 = ("FALSIFIED by " + ", ".join(falsified)) if falsified else (("HELD for " + ", ".join(held)) if held else "not evaluable")
                if nev: p1 += "; not evaluable for " + ", ".join(nev)
                P["P1 no RL arm 'RL better' on acc_custom"] = p1
                cc = [rl for rl in C["paired"] if "creditC" in rl]
                if cc: P["P2 credit-C truncation % >= base"] = "HELD" if C["arms"][cc[0]]["gates"]["trunc_pct"] >= C["arms"][a.base]["gates"]["trunc_pct"] else "NOT HELD"
                P["P3 every arm: acc recognisable > acc custom"] = "HELD" if all((C["arms"][arm]["summary"]["acc_recognisable"]["mean"] or 0) > (C["arms"][arm]["summary"]["acc_custom"]["mean"] or 0) for arm in present) else "NOT HELD"
            C["predictions"] = P
            L += ["", "Directional predictions: " + "; ".join(f"{k}: **{v}**" for k, v in P.items())]
        L.append("")
        out["conditions"][f"think{think}/{mode}"] = C
# ---- confirmatory dialect per RL arm (pre-registration §5.5): ON primary; ON confounded -> OFF if clean; both confounded -> no decision
conf_out = {}; CL = ["## Confirmatory dialect per RL arm (pre-registration §5.5: ON is primary; ON confounded -> OFF becomes confirmatory if it passes all gates; both confounded -> no decision)", ""]
for rl in RL:
    def state(key):
        C = out["conditions"].get(key)
        if not C or rl not in C["paired"]: return "absent", []
        if C["incomplete"]: return "incomplete", []
        g = C["paired"][rl]["confound_gates_tripped"]
        return ("confounded" if g else "clean"), g
    s_on, g_on = state("thinkon/nohint"); s_off, g_off = state("thinkoff/nohint")
    if s_on == "clean": dialect, why = "on", "ON (primary) passes all gates"
    elif s_on == "confounded":
        if s_off == "clean": dialect, why = "off", f"ON confounded ({','.join(g_on)}); OFF passes all gates"
        elif s_off == "confounded": dialect, why = None, f"NO DECISION: both dialects confounded (ON {','.join(g_on)}; OFF {','.join(g_off)})"
        elif s_off == "incomplete": dialect, why = None, f"pending: ON confounded ({','.join(g_on)}); OFF incomplete"
        else: dialect, why = None, f"pending: ON confounded ({','.join(g_on)}); OFF not run (OFF would be confirmatory if it passes all gates)"
    elif s_on == "incomplete": dialect, why = None, "pending: ON incomplete"
    else: dialect, why = None, "pending: ON (primary) not run" + ("; OFF present but secondary only" if s_off != "absent" else "")
    dec = {}
    if dialect:
        P = out["conditions"][f"think{dialect}/nohint"]["paired"][rl]
        dec = {k: P[k]["decision"] for k in ("acc_custom (PRIMARY)", "acc_overall (CO-PRIMARY)") if k in P}
    conf_out[rl] = {"confirmatory_dialect": dialect, "reason": why, "decisions": dec}
    CL.append(f"CONFIRMATORY {rl}: dialect={dialect.upper() if dialect else 'none'} — {why}" + (f" -> {'; '.join(f'{k}: {v}' for k, v in dec.items())}" if dec else ""))
if out["conditions"]: L += CL + [""]
out["hint_trigger"] = hint_trigger; out["confirmatory"] = conf_out
json.dump(out, open(os.path.join(a.root, "CIPHERBANK_SUMMARY.json"), "w"), indent=1, default=str)
open(os.path.join(a.root, "CIPHERBANK_SUMMARY.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
