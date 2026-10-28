#!/usr/bin/env python3
"""
cb_driver.py — CipherBank (Leey21/CipherBank @ 05b450c, HF yu0226/CipherBank @ d2f8f7cc) item runner for ONE arm.
Called by launch_cipherbank.sh; reuses the benchmark's own prompt template, <result> extraction and scoring by importing
them from the pinned repo (prompt.get_prompt, utils.tools.{OpenAILLM, extract_result_sentences, compare_strings,
levenshtein_similarity}); only the loop (threads, resume, STOP file, call log, infra re-run) is ours.

  cb_driver.py build-data --repo REPO --hf-dir DIR [--out REPO/data/test_hf.jsonl]
        merge the 9 pinned HF files (one per cipher, row-aligned) into the repo's one-row-per-plaintext format
  cb_driver.py run --repo REPO --root ROOT --think on|off --mode nohint|hint --arm base --name SERVED --port P
        --reps 1,2 --par 32 --max-tokens N [--rows all|N] [--data FILE] [--no-save-outputs]
  cb_driver.py status --root ROOT [--think on|off] [--mode nohint|hint] [--rows all|N]
  cb_driver.py seedcheck --root ROOT --think on|off --mode nohint|hint --arm base --name SERVED --port P [--n 2] [--rep 1]
        re-send N complete items of that arm with their recorded seed and diff the content against outputs.ndjson:
        tells whether the server honours `seed` (identical -> seed honoured; different -> reps are plain resamples);
        the verdict is written to ROOT/provenance/seedcheck_<arm>_think<T>_<ts>.json (nothing is scored)

Layout  ROOT/think<T>/<mode>/<served-name>/rep<k>/{results.ndjson, calls.ndjson, outputs.ndjson}
  results.ndjson  one line per finished (item, rep) attempt: scores + gate fields + max_tokens (lean; last line per item wins)
  calls.ndjson    one line per DRIVER ATTEMPT (each = up to 6 HTTP requests: OpenAI SDK max_retries=2 x the benchmark wrapper's
                  max_num_retries=2), incl. failed ones: ts, arm, item, rep, attempt, finish_reason, tokens, latency, wrapper_attempts, error
  outputs.ndjson  raw content + reasoning_content per finished item (forensics; big under thinking ON)
Item id = "<Cipher>/<row:03d>" (row = 0-based line in data/test_hf.jsonl); 262 rows x 9 ciphers = 2,358 items.
Seed sent per request = 1_000_000*rep + 10*row + cipher_index, identical across arms so repetitions are paired by (item, rep).
(SGLang 0.5.19 accepts `seed`; whether it is honoured deterministically is checked at smoke time by `seedcheck`, not assumed.)
Completion rule: an item is complete when its LAST results record has error == null AND was produced under the current
--max-tokens (a record run under another budget is re-run, so a root never mixes budgets silently). A record with error != null
(endpoint failure after the in-run re-run) is scored 0 and counted per arm by the summarizer, and is re-run by the next invocation.
A DRIVER-ERROR (exception while building the prompt or scoring) writes NO record: the item stays missing, the run exits 6 and the
summarizer prints MISSING + INCOMPLETE (no decision) until it is complete.
STOP: ROOT/STOP checked before every item; a stopped run exits 5 (nothing partial is written for unstarted items).
"""
import argparse, hashlib, json, os, queue, re, socket, sys, threading, time

CIPHERS = ["Rot13", "Atbash", "Polybius", "Vigenere", "Reverse", "SwapPairs", "ParityShift", "DualAvgCode", "WordShift"]
FAMILY = {"Rot13": "substitution", "Atbash": "substitution", "Polybius": "substitution", "Vigenere": "substitution",
          "Reverse": "transposition", "SwapPairs": "transposition",
          "ParityShift": "custom", "DualAvgCode": "custom", "WordShift": "custom"}
TIER = {"Rot13": "T1", "Reverse": "T1", "Atbash": "T2", "WordShift": "T2", "DualAvgCode": "T3", "Polybius": "T3",
        "ParityShift": "T4", "SwapPairs": "T4", "Vigenere": "T5"}          # paper §2.2, five difficulty tiers
RECOGNISABLE = {"Rot13", "Atbash", "Reverse"}                            # pretraining-recognisable: report, do not headline
RESULT_RE = re.compile(r"<result>(.*?)</result>", re.DOTALL)              # same regex as utils.tools.extract_result_sentences

def ts(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()
def read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass
    return out

# ----------------------------------------------------------------------------------------------------------- build-data
def build_data(a):
    per = {}
    for c in CIPHERS:
        p = os.path.join(a.hf_dir, f"{c}.jsonl")
        per[c] = read_jsonl(p)
    n = {c: len(v) for c, v in per.items()}
    assert len(set(n.values())) == 1, f"HF files have different row counts: {n}"
    rows = []
    for i in range(n[CIPHERS[0]]):
        base = per[CIPHERS[0]][i]
        row = {"domain": base["domain"], "sub_domain": base["sub_domain"], "plaintext": base["plaintext"]}
        for c in CIPHERS:
            r = per[c][i]
            assert r["plaintext"] == row["plaintext"] and r["domain"] == row["domain"] and r["sub_domain"] == row["sub_domain"], f"row {i} not aligned across HF files ({c})"
            row[c] = r[c]
        rows.append(row)
    out = a.out or os.path.join(a.repo, "data", "test_hf.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"build-data: {len(rows)} rows x {len(CIPHERS)} ciphers = {len(rows)*len(CIPHERS)} items -> {out} md5 {md5(out)}")

# ----------------------------------------------------------------------------------------------------------- run
def load_last_records(path):
    last = {}
    if os.path.exists(path):
        for r in read_jsonl(path):
            last[r["item"]] = r
    return last

def run(a):
    a.root = os.path.abspath(a.root); a.repo = os.path.abspath(a.repo)
    if a.data: a.data = os.path.abspath(a.data)
    os.chdir(a.repo); sys.path.insert(0, a.repo)
    import prompt as cb_prompt                                            # loads data/shot_case.jsonl relative to cwd
    from utils.tools import OpenAILLM, extract_result_sentences, compare_strings, levenshtein_similarity
    data_path = a.data or os.path.join(a.repo, "data", "test_hf.jsonl")
    data = read_jsonl(data_path)
    rows = list(range(len(data))) if a.rows == "all" else list(range(min(int(a.rows), len(data))))
    reps = [int(x) for x in a.reps.split(",") if x]
    stop_file = os.path.join(a.root, "STOP")
    base_url = f"http://127.0.0.1:{a.port}/v1"
    llm = OpenAILLM(model_name=a.name, base_url=base_url, api_key="x")
    is_hint = a.mode == "hint"
    lock = threading.Lock()
    stats = {"done": 0, "skipped": 0, "errors": 0, "started": 0, "stopped": False, "driver_errors": 0, "rerun_other_budget": 0}
    files = {}
    for rep in reps:
        d = os.path.join(a.root, f"think{a.think}", a.mode, a.name, f"rep{rep}")
        os.makedirs(d, exist_ok=True)
        files[rep] = {"results": os.path.join(d, "results.ndjson"), "calls": os.path.join(d, "calls.ndjson"), "outputs": os.path.join(d, "outputs.ndjson"),
                      "last": load_last_records(os.path.join(d, "results.ndjson"))}
    jobs = queue.Queue()
    n_jobs = 0
    for rep in reps:                                                       # rep-major so rep 1 finishes first
        for row in rows:
            for ci, c in enumerate(CIPHERS):
                item = f"{c}/{row:03d}"
                last = files[rep]["last"].get(item)
                if last is not None and last.get("error") is None:
                    if last.get("max_tokens") == a.max_tokens:
                        stats["skipped"] += 1; continue
                    stats["rerun_other_budget"] += 1                       # complete, but under another max_tokens -> re-run
                jobs.put((rep, row, ci, c)); n_jobs += 1
    print(f"{ts()} run arm={a.arm} name={a.name} port={a.port} think={a.think} mode={a.mode} reps={reps} rows={len(rows)} "
          f"items/rep={len(rows)*len(CIPHERS)} to_run={n_jobs} skipped_complete={stats['skipped']} rerun_other_max_tokens={stats['rerun_other_budget']} "
          f"par={a.par} max_tokens={a.max_tokens} data={data_path} md5={md5(data_path)}", flush=True)

    def append(path, rec):
        with lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run_item(rep, row, ci, c):
        item = f"{c}/{row:03d}"; rec_row = data[row]
        system_prompt, user_prompt = cb_prompt.get_prompt(format_type="markdown", cipher_type=c, few_shot=3,
                                                          ciphertext=rec_row[c], is_hint=is_hint, number=False)
        seed = 1_000_000 * rep + 10 * row + ci
        full = None; err = None; attempts = 0
        for attempt in (1, 2):                                             # pre-registered: one in-run re-run of endpoint failures
            attempts = attempt
            t0 = time.time()
            try:
                full = llm.generate(user_prompt, system_prompt, max_completion_tokens=a.max_tokens, seed=seed, return_full=True)
                err = None
            except Exception as e:                                         # RuntimeError after the wrapper's own retries, or transport errors
                err = f"{type(e).__name__}: {str(e)[:300]}"; full = None
            call = {"ts": ts(), "arm": a.arm, "served": a.name, "item": item, "rep": rep, "attempt": attempt, "seed": seed, "max_tokens": a.max_tokens,
                    "finish_reason": full.get("finish_reason") if full else None, "matched_stop": full.get("matched_stop") if full else None,
                    "prompt_tokens": full.get("prompt_tokens") if full else None, "completion_tokens": full.get("completion_tokens") if full else None,
                    "reasoning_tokens": full.get("reasoning_tokens") if full else None,
                    "latency_s": round(full["latency_s"], 3) if full else round(time.time() - t0, 3),
                    "wrapper_attempts": full.get("attempts") if full else None, "error": err}   # wrapper tries (1-2), each <= 3 HTTP requests
            append(files[rep]["calls"], call)
            if err is None: break
            print(f"INFRA {ts()} {a.name} rep{rep} {item} attempt={attempt}: {err}", flush=True)
            if attempt == 1: time.sleep(float(os.getenv("CB_RETRY_SLEEP", "10")))
        content = (full or {}).get("response")
        text = content if isinstance(content, str) else ""
        sentence = extract_result_sentences(text)                          # upstream: last <result>…</result>, else whole text
        has_tag = RESULT_RE.search(text) is not None
        is_correct = bool(compare_strings(sentence, rec_row["plaintext"])) if text else False
        lev = float(levenshtein_similarity(sentence, rec_row["plaintext"])) if text else 0.0
        reasoning = (full or {}).get("reasoning_content")
        res = {"item": item, "cipher": c, "row": row, "family": FAMILY[c], "tier": TIER[c], "recognisable": c in RECOGNISABLE,
               "rep": rep, "arm": a.arm, "served": a.name, "think": a.think, "mode": a.mode, "seed": seed, "max_tokens": a.max_tokens,
               "domain": rec_row["domain"], "sub_domain": rec_row["sub_domain"], "plaintext_len": len(rec_row["plaintext"]),
               "prediction": sentence if has_tag else None, "is_correct": is_correct, "levenshtein": lev,
               "has_result_tag": has_tag, "empty_answer": (sentence.strip() == "") if has_tag else (text.strip() == ""),
               "finish_reason": (full or {}).get("finish_reason"), "matched_stop": (full or {}).get("matched_stop"),
               "prompt_tokens": (full or {}).get("prompt_tokens"), "completion_tokens": (full or {}).get("completion_tokens"),
               "reasoning_tokens": (full or {}).get("reasoning_tokens"), "response_chars": len(text),
               "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else 0,
               "latency_s": round(full["latency_s"], 3) if full else None, "attempts": attempts, "error": err, "ts": ts()}
        append(files[rep]["results"], res)
        if a.save_outputs and full is not None:
            append(files[rep]["outputs"], {"item": item, "rep": rep, "served": a.name, "response": content, "reasoning_content": reasoning})
        with lock:
            stats["done"] += 1
            if err is not None: stats["errors"] += 1
            n = stats["done"]
        tag = "FAILED" if err is not None else "end  "
        print(f"{tag} {ts()} {a.name} rep{rep} {item} correct={int(is_correct)} lev={lev:.3f} tag={int(has_tag)} finish={res['finish_reason']} "
              f"ctok={res['completion_tokens']} rtok={res['reasoning_tokens']} lat={res['latency_s']} attempts={attempts}" + (f" error={err}" if err else ""), flush=True)
        if n % 50 == 0:
            print(f"{ts()} progress {a.name}: done={n}/{n_jobs} errors={stats['errors']}", flush=True)

    def worker():
        while True:
            if os.path.exists(stop_file):
                with lock: stats["stopped"] = True
                return
            try: job = jobs.get_nowait()
            except queue.Empty: return
            with lock: stats["started"] += 1
            try: run_item(*job)
            except Exception as e:                                         # never lose the pool to one bad item; but never hide it either
                with lock: stats["driver_errors"] += 1
                print(f"DRIVER-ERROR {ts()} {a.name} rep{job[0]} {job[3]}/{job[1]:03d}: {type(e).__name__}: {e}", flush=True)
            finally: jobs.task_done()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, a.par))]
    for t in threads: t.start()
    for t in threads: t.join()
    left = jobs.qsize()
    print(f"{ts()} run done arm={a.arm} name={a.name} think={a.think} mode={a.mode}: started={stats['started']} done={stats['done']} "
          f"errors_scored_0={stats['errors']} driver_errors_no_record={stats['driver_errors']} skipped_complete={stats['skipped']} not_started={left}"
          + (" STOPPED (STOP file present)" if stats["stopped"] else ""), flush=True)
    if stats["driver_errors"]:
        print(f"{ts()} DRIVER-ERRORS={stats['driver_errors']}: these items have NO results record (missing, not scored); fix the cause and re-run -> exit 6", flush=True)
        sys.exit(6)
    sys.exit(5 if stats["stopped"] else 0)

# ----------------------------------------------------------------------------------------------------------- status
def status(a):
    import glob
    n_items = (262 if a.rows == "all" else int(a.rows)) * len(CIPHERS)
    pat = os.path.join(a.root, f"think{a.think or '*'}", a.mode or "*", "*", "rep*", "results.ndjson")
    found = sorted(glob.glob(pat))
    if not found: print(f"status: no results under {pat}"); return
    for p in found:
        parts = p.split(os.sep); think, mode, name, rep = parts[-5], parts[-4], parts[-3], parts[-2]
        last = load_last_records(p)
        complete = sum(1 for r in last.values() if r.get("error") is None); errored = len(last) - complete
        budgets = sorted({str(r.get("max_tokens")) for r in last.values() if r.get("error") is None})
        calls = os.path.join(os.path.dirname(p), "calls.ndjson")
        n_calls = sum(1 for _ in open(calls, encoding="utf-8")) if os.path.exists(calls) else 0
        missing = n_items - len(last)
        print(f"{think} {mode} {name} {rep}: complete={complete}/{n_items} errored(last record)={errored} missing={missing} "
              f"driver_attempts={n_calls} max_tokens_seen={','.join(budgets) or '-'}")
        if missing > 0 or errored > 0: print(f"  WARNING {name} {rep}: {missing} missing (no record) + {errored} errored -> re-run the same play command (missing after a re-run = DRIVER-ERROR in the log)")
        if len(budgets) > 1: print(f"  WARNING {name} {rep}: mixed max_tokens {budgets} -> re-run under the intended budget (the driver re-runs items completed under another budget)")

# ----------------------------------------------------------------------------------------------------------- seedcheck
def seedcheck(a):
    """re-send N complete items with their recorded seed; identical content <=> the server honours `seed`. Writes provenance, scores nothing."""
    a.root = os.path.abspath(a.root); a.repo = os.path.abspath(a.repo)
    os.chdir(a.repo); sys.path.insert(0, a.repo)
    import prompt as cb_prompt
    from utils.tools import OpenAILLM
    d = os.path.join(a.root, f"think{a.think}", a.mode, a.name, f"rep{a.rep}")
    last = load_last_records(os.path.join(d, "results.ndjson"))
    outs = {r["item"]: r for r in read_jsonl(os.path.join(d, "outputs.ndjson"))} if os.path.exists(os.path.join(d, "outputs.ndjson")) else {}
    data = read_jsonl(a.data or os.path.join(a.repo, "data", "test_hf.jsonl"))
    items = [it for it, r in sorted(last.items()) if r.get("error") is None and it in outs and outs[it].get("response")][: a.n]
    if not items: print("seedcheck: no complete items with saved outputs under", d); sys.exit(2)
    llm = OpenAILLM(model_name=a.name, base_url=f"http://127.0.0.1:{a.port}/v1", api_key="x")
    res = []
    for it in items:
        r = last[it]; c, row = r["cipher"], int(r["row"])
        sp, up = cb_prompt.get_prompt(format_type="markdown", cipher_type=c, few_shot=3, ciphertext=data[row][c], is_hint=(a.mode == "hint"), number=False)
        full = llm.generate(up, sp, max_completion_tokens=int(r["max_tokens"]), seed=int(r["seed"]), return_full=True)
        same = (full.get("response") or "") == (outs[it].get("response") or "")
        res.append({"item": it, "seed": r["seed"], "max_tokens": r["max_tokens"], "identical_content": same, "finish_reason": full.get("finish_reason"),
                    "orig_completion_tokens": r.get("completion_tokens"), "new_completion_tokens": full.get("completion_tokens")})
        print(f"seedcheck {a.name} {it} seed={r['seed']}: {'IDENTICAL' if same else 'DIFFERENT'} (ctok {r.get('completion_tokens')} -> {full.get('completion_tokens')})", flush=True)
    n_same = sum(x["identical_content"] for x in res)
    verdict = "seed honoured (deterministic per seed)" if n_same == len(res) else ("seed NOT honoured (reps are plain resamples)" if n_same == 0 else "MIXED (some identical, some not)")
    out = {"bench": "cipherbank", "check": "seedcheck", "ts": ts(), "arm": a.arm, "served": a.name, "think": a.think, "mode": a.mode, "rep": a.rep,
           "n": len(res), "n_identical": n_same, "verdict": verdict, "items": res}
    os.makedirs(os.path.join(a.root, "provenance"), exist_ok=True)
    p = os.path.join(a.root, "provenance", f"seedcheck_{a.arm}_think{a.think}_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"seedcheck verdict: {verdict} ({n_same}/{len(res)} identical) -> {p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build-data"); b.add_argument("--repo", required=True); b.add_argument("--hf-dir", required=True); b.add_argument("--out")
    r = sp.add_parser("run")
    r.add_argument("--repo", required=True); r.add_argument("--root", required=True); r.add_argument("--think", required=True, choices=["on", "off"])
    r.add_argument("--mode", default="nohint", choices=["nohint", "hint"]); r.add_argument("--arm", required=True); r.add_argument("--name", required=True)
    r.add_argument("--port", required=True, type=int); r.add_argument("--reps", default="1,2"); r.add_argument("--par", type=int, default=32)
    r.add_argument("--max-tokens", type=int, required=True); r.add_argument("--rows", default="all"); r.add_argument("--data")
    r.add_argument("--no-save-outputs", dest="save_outputs", action="store_false")
    s = sp.add_parser("status"); s.add_argument("--root", required=True); s.add_argument("--think"); s.add_argument("--mode"); s.add_argument("--rows", default="all")
    k = sp.add_parser("seedcheck")
    k.add_argument("--repo", required=True); k.add_argument("--root", required=True); k.add_argument("--think", required=True, choices=["on", "off"])
    k.add_argument("--mode", default="nohint", choices=["nohint", "hint"]); k.add_argument("--arm", required=True); k.add_argument("--name", required=True)
    k.add_argument("--port", required=True, type=int); k.add_argument("--n", type=int, default=2); k.add_argument("--rep", type=int, default=1); k.add_argument("--data")
    A = ap.parse_args()
    {"build-data": build_data, "run": run, "status": status, "seedcheck": seedcheck}[A.cmd](A)
