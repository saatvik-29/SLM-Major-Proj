"""
=============================================================================
  CUSTOMER SUPPORT SYSTEM - LIVE API TEST RUNNER
  Server  : http://localhost:8000
  Outputs : api_test_results.json   <- full structured results
            api_test_results.csv    <- spreadsheet-friendly summary
  Reads   : test_questions.json     <- dynamic question bank
=============================================================================
"""

import sys
import io
# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
import json
import csv
import os
import time
from datetime import datetime

BASE_URL      = "http://localhost:8000"
PROJECT_ROOT  = os.path.dirname(__file__)
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, "test_questions.json")
JSON_RESULTS   = os.path.join(PROJECT_ROOT, "api_test_results.json")
CSV_RESULTS    = os.path.join(PROJECT_ROOT, "api_test_results.csv")

# ── sample document (embedded; no external file needed) ──────────────────────
SAMPLE_DOC_TEXT = """
CUSTOMER SUPPORT POLICY v1.0
==============================

Refund Policy:
  Customers may request a refund within 30 days of purchase.
  Refunds are processed within 5-7 business days.
  Digital products are non-refundable once downloaded.

Password Reset:
  Users can reset their password by clicking Forgot Password on the login page.
  A reset link is sent to the registered email address.
  The link expires after 24 hours.

Billing:
  We accept Visa, Mastercard, and PayPal.
  Invoices are generated on the 1st of each month.
  For billing disputes contact billing@support.example.com

Return Policy:
  Items must be returned in original packaging.
  Return shipping is the customer's responsibility.
  Store credit is issued within 3 business days after inspection.

Contact:
  Email: support@example.com
  Phone: 1-800-555-0100 (Mon-Fri 9AM-6PM EST)
  Live chat available 24/7 on our website.
""".strip()

SAMPLE_DOC_PATH = os.path.join(PROJECT_ROOT, "_tmp_seed.txt")

# ─────────────────────────────────────────────────────────────────────────────
#  RESULT STORE
# ─────────────────────────────────────────────────────────────────────────────
results    = []
pass_count = 0
fail_count = 0

RUN_META = {
    "server": BASE_URL,
    "run_date": "",
    "total": 0,
    "passed": 0,
    "failed": 0,
    "duration_seconds": 0,
}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def run_test(tc_id: str, name: str, fn, session: requests.Session, tags: list = None):
    global pass_count, fail_count
    print(f"  {tc_id}: {name} ...", end=" ", flush=True)
    start = time.time()
    try:
        passed, inp, out, note = fn(session)
        elapsed = round(time.time() - start, 3)
        status  = "PASS" if passed else "FAIL"
        if passed: pass_count += 1
        else:      fail_count += 1
        print(f"[{status}] ({elapsed}s)")
        results.append({
            "id": tc_id, "name": name,
            "tags": tags or [],
            "status": status,
            "input": inp,
            "output": out,
            "note": note,
            "elapsed_seconds": elapsed,
        })
    except Exception as e:
        elapsed = round(time.time() - start, 3)
        fail_count += 1
        print(f"[ERROR] ({elapsed}s)")
        results.append({
            "id": tc_id, "name": name,
            "tags": tags or [],
            "status": "ERROR",
            "input": "N/A",
            "output": str(e),
            "note": "Unhandled exception",
            "elapsed_seconds": elapsed,
        })


def seed_index(session: requests.Session):
    """Make sure the index has data before query tests."""
    with open(SAMPLE_DOC_PATH, "rb") as f:
        session.post(f"{BASE_URL}/v1/documents/upload",
                     files=[("files", ("seed.txt", f, "text/plain"))])


# ─────────────────────────────────────────────────────────────────────────────
#  TEST FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── /health ──────────────────────────────────────────────────────────────────
def t_health_ok(s):
    r = s.get(f"{BASE_URL}/health")
    b = r.json()
    ok = r.status_code == 200 and b.get("status") == "ok"
    return ok, "GET /health", json.dumps(b, indent=2), "status==ok" if ok else f"Got: {b}"

def t_health_components(s):
    r = s.get(f"{BASE_URL}/health")
    comps = r.json().get("components", {})
    ok = "rag" in comps and "slm" in comps
    return ok, "GET /health -> components", json.dumps(comps, indent=2), \
           "rag+slm present" if ok else "Missing components"

def t_health_slm_ready(s):
    r = s.get(f"{BASE_URL}/health")
    slm = r.json().get("components", {}).get("slm", "")
    ok = slm == "ready"
    return ok, "GET /health -> slm field", f"slm: {slm}", \
           "SLM ready" if ok else f"SLM: {slm}"

def t_health_content_type(s):
    r = s.get(f"{BASE_URL}/health")
    ct = r.headers.get("content-type", "")
    ok = "application/json" in ct
    return ok, "GET /health -> Content-Type header", ct, \
           "application/json" if ok else f"Wrong: {ct}"

# ── /v1/documents/upload ─────────────────────────────────────────────────────
def t_upload_txt(s):
    with open(SAMPLE_DOC_PATH, "rb") as f:
        r = s.post(f"{BASE_URL}/v1/documents/upload",
                   files=[("files", ("policy.txt", f, "text/plain"))])
    b = r.json()
    ok = r.status_code == 200 and b.get("status") == "success"
    return ok, "POST /v1/documents/upload  file=policy.txt", json.dumps(b, indent=2), \
           f"chunks={b.get('chunks_indexed')}"

def t_upload_chunks_positive(s):
    with open(SAMPLE_DOC_PATH, "rb") as f:
        r = s.post(f"{BASE_URL}/v1/documents/upload",
                   files=[("files", ("policy2.txt", f, "text/plain"))])
    chunks = r.json().get("chunks_indexed", 0)
    ok = chunks > 0
    return ok, "POST /v1/documents/upload -> chunks_indexed > 0", \
           f"chunks_indexed = {chunks}", "OK" if ok else "No chunks!"

def t_upload_filename_recorded(s):
    with open(SAMPLE_DOC_PATH, "rb") as f:
        r = s.post(f"{BASE_URL}/v1/documents/upload",
                   files=[("files", ("myfile.txt", f, "text/plain"))])
    fps = r.json().get("files_processed", [])
    ok  = "myfile.txt" in fps
    return ok, "POST /v1/documents/upload -> filename in response", str(fps), \
           "Filename recorded" if ok else "Not found"

def t_upload_multiple(s):
    content = SAMPLE_DOC_TEXT.encode()
    r = s.post(f"{BASE_URL}/v1/documents/upload",
               files=[("files", ("a.txt", content, "text/plain")),
                      ("files", ("b.txt", content, "text/plain"))])
    b   = r.json()
    fps = b.get("files_processed", [])
    ok  = r.status_code == 200 and len(fps) == 2
    return ok, "POST /v1/documents/upload  files=[a.txt, b.txt]", \
           json.dumps(b, indent=2), f"Processed: {fps}"

def t_upload_no_file(s):
    r  = s.post(f"{BASE_URL}/v1/documents/upload")
    ok = r.status_code == 422
    return ok, "POST /v1/documents/upload  (no files - validation)", \
           f"HTTP {r.status_code}: {r.text[:200]}", \
           "422 as expected" if ok else f"Unexpected {r.status_code}"

# ── /v1/query ─────────────────────────────────────────────────────────────────
def make_query_test(question: str, keywords: list, label: str = None):
    """Factory: returns a test function for a given question."""
    def _test(s):
        r    = s.post(f"{BASE_URL}/v1/query", json={"question": question})
        b    = r.json()
        ans  = b.get("answer", "").lower()
        ok   = r.status_code == 200 and (
                    not keywords or
                    any(k.lower() in ans for k in keywords)
               )
        input_repr  = f'POST /v1/query  question="{question}"'
        output_repr = json.dumps(b, indent=2)
        note = (f"Keyword hit: {[k for k in keywords if k.lower() in ans]}"
                if ok and keywords else
                f"No keyword match — answer: {ans[:120]}" if not ok else "Responded OK")
        return ok, input_repr, output_repr, note
    return _test

def t_query_empty(s):
    r  = s.post(f"{BASE_URL}/v1/query", json={"question": ""})
    ok = r.status_code in (200, 422)
    return ok, 'POST /v1/query  question="" (empty)', \
           f"HTTP {r.status_code}: {r.text[:300]}", \
           "Handled empty" if ok else "Crash on empty"

def t_query_missing_field(s):
    r  = s.post(f"{BASE_URL}/v1/query", json={})
    ok = r.status_code == 422
    return ok, "POST /v1/query  payload={} (missing field)", \
           f"HTTP {r.status_code}: {r.text[:300]}", \
           "422 as expected" if ok else f"Got {r.status_code}"

def t_query_json_schema(s):
    r = s.post(f"{BASE_URL}/v1/query", json={"question": "refund?"})
    b = r.json()
    ok = "answer" in b and "sources" in b
    return ok, "POST /v1/query -> verify JSON schema", json.dumps(b, indent=2), \
           "Schema OK {answer, sources}" if ok else f"Keys: {list(b.keys())}"

def t_query_response_time(s):
    t0  = time.time()
    r   = s.post(f"{BASE_URL}/v1/query", json={"question": "What is the refund policy?"})
    ela = time.time() - t0
    ok  = r.status_code == 200 and ela < 120
    return ok, "POST /v1/query + measure response time (limit 120s)", \
           f"Response time: {ela:.3f}s  HTTP {r.status_code}", \
           f"{'OK' if ela < 120 else 'SLOW'} ({ela:.1f}s)"

def t_query_sources_list(s):
    r   = s.post(f"{BASE_URL}/v1/query", json={"question": "refund?"})
    src = r.json().get("sources", None)
    ok  = isinstance(src, list)
    return ok, "POST /v1/query -> sources field is list", \
           f"sources = {src}", "sources is list" if ok else "sources missing"

# ── /v1/reset ─────────────────────────────────────────────────────────────────
def t_reset_ok(s):
    r  = s.post(f"{BASE_URL}/v1/reset")
    b  = r.json()
    ok = r.status_code == 200 and b.get("status") == "success"
    return ok, "POST /v1/reset", json.dumps(b, indent=2), \
           "Reset OK" if ok else f"Got: {b}"

def t_reset_empty_query(s):
    s.post(f"{BASE_URL}/v1/reset")
    r   = s.post(f"{BASE_URL}/v1/query", json={"question": "refund policy"})
    ans = r.json().get("answer", "").lower()
    ok  = r.status_code == 200 and "don't know" in ans
    return ok, "POST /v1/reset -> POST /v1/query on empty index", \
           json.dumps(r.json(), indent=2), \
           "Returns dont-know on empty" if ok else f"Unexpected: {ans}"

def t_reset_idempotent(s):
    s.post(f"{BASE_URL}/v1/reset")
    r2 = s.post(f"{BASE_URL}/v1/reset")
    ok = r2.status_code == 200
    return ok, "POST /v1/reset x2 (idempotency)", \
           f"2nd reset HTTP {r2.status_code}: {r2.text[:200]}", \
           "Idempotent" if ok else "2nd reset failed"

def t_upload_after_reset(s):
    s.post(f"{BASE_URL}/v1/reset")
    with open(SAMPLE_DOC_PATH, "rb") as f:
        r = s.post(f"{BASE_URL}/v1/documents/upload",
                   files=[("files", ("fresh.txt", f, "text/plain"))])
    b  = r.json()
    ok = r.status_code == 200 and b.get("chunks_indexed", 0) > 0
    return ok, "POST /v1/reset -> POST /v1/documents/upload", \
           json.dumps(b, indent=2), "Re-upload after reset works" if ok else "Failed"

# ── CORS ──────────────────────────────────────────────────────────────────────
def t_cors(s):
    r   = s.options(f"{BASE_URL}/health",
                    headers={"Origin": "http://localhost:3000",
                             "Access-Control-Request-Method": "GET"})
    acao = r.headers.get("access-control-allow-origin", "")
    ok   = acao in ("*", "http://localhost:3000")
    return ok, "OPTIONS /health (CORS preflight)", \
           f"access-control-allow-origin: {acao}", \
           "CORS header present" if ok else "Missing CORS"

# ── E2E ───────────────────────────────────────────────────────────────────────
def t_e2e(s):
    s.post(f"{BASE_URL}/v1/reset")
    with open(SAMPLE_DOC_PATH, "rb") as f:
        s.post(f"{BASE_URL}/v1/documents/upload",
               files=[("files", ("e2e.txt", f, "text/plain"))])
    r   = s.post(f"{BASE_URL}/v1/query",
                 json={"question": "How long does a refund take?"})
    b   = r.json()
    ans = b.get("answer", "").lower()
    ok  = r.status_code == 200 and any(k in ans for k in ["day","5","7","business"])
    return ok, "E2E: reset -> upload -> query refund timeline", \
           json.dumps(b, indent=2), "Timeline in answer" if ok else f"Got: {ans}"


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────
STATIC_TESTS = [
    # Health
    ("TC_API_01", "Health returns 200 + status=ok",           t_health_ok,           ["health"]),
    ("TC_API_02", "Health includes components object",         t_health_components,   ["health"]),
    ("TC_API_03", "Health shows SLM as ready",                 t_health_slm_ready,    ["health","slm"]),
    ("TC_API_04", "Health returns JSON content-type",          t_health_content_type, ["health"]),
    # Upload
    ("TC_API_05", "Upload .txt document successfully",         t_upload_txt,          ["upload"]),
    ("TC_API_06", "Upload chunks_indexed > 0",                 t_upload_chunks_positive, ["upload"]),
    ("TC_API_07", "Upload filename in response",               t_upload_filename_recorded, ["upload"]),
    ("TC_API_08", "Upload multiple files in one request",      t_upload_multiple,     ["upload"]),
    ("TC_API_09", "Upload with no files -> 422",               t_upload_no_file,      ["upload","validation"]),
    # Query - structural
    ("TC_API_10", "Query: JSON schema has answer+sources",     t_query_json_schema,   ["query","schema"]),
    ("TC_API_11", "Query: sources field is a list",            t_query_sources_list,  ["query","schema"]),
    ("TC_API_12", "Query: response time < 120s",               t_query_response_time, ["query","perf"]),
    ("TC_API_13", "Query: empty question handled",             t_query_empty,         ["query","edge"]),
    ("TC_API_14", "Query: missing field -> 422",               t_query_missing_field, ["query","validation"]),
    # Reset
    ("TC_API_15", "Reset returns status=success",              t_reset_ok,            ["reset"]),
    ("TC_API_16", "Reset + query on empty index -> dont-know", t_reset_empty_query,   ["reset","query"]),
    ("TC_API_17", "Reset is idempotent (called twice)",        t_reset_idempotent,    ["reset"]),
    ("TC_API_18", "Upload after reset works",                  t_upload_after_reset,  ["reset","upload"]),
    # CORS
    ("TC_API_19", "CORS preflight returns allow-origin",       t_cors,                ["cors"]),
    # E2E
    ("TC_API_20", "E2E: reset -> upload -> query",             t_e2e,                 ["e2e"]),
]


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC TESTS FROM test_questions.json
# ─────────────────────────────────────────────────────────────────────────────
def load_question_tests() -> list:
    """Load test_questions.json (only 'question' field) and build test tuples."""
    if not os.path.exists(QUESTIONS_FILE):
        print(f"  [WARN] {QUESTIONS_FILE} not found, skipping dynamic tests.")
        return []

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    tests = []
    for i, q in enumerate(questions, start=1):
        question = q.get("question", "")
        tc_id    = f"TC_Q_{i:03d}"
        label    = question[:60] if question.strip() else "(empty)"
        name     = label
        fn       = make_query_test(question, [])   # no keyword checking — just record answer
        tags     = ["dynamic"]
        tests.append((tc_id, name, fn, tags))

    return tests


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT — JSON
# ─────────────────────────────────────────────────────────────────────────────
def save_json():
    """Save only {question, answer} pairs — one per dynamic test run."""
    qa_pairs = []
    for r in results:
        if "dynamic" in r.get("tags", []):
            # extract question from the stored input string
            inp = r.get("input", "")
            question = ""
            if 'question="' in inp:
                question = inp.split('question="', 1)[1].rsplit('"', 1)[0]
            # extract answer from the stored output JSON
            answer = ""
            try:
                out_obj = json.loads(r.get("output", "{}"))
                answer  = out_obj.get("answer", "")
            except Exception:
                answer = r.get("output", "")
            qa_pairs.append({"question": question, "answer": answer})

    with open(JSON_RESULTS, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"  JSON -> {JSON_RESULTS}")


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORT — CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_csv():
    fields = ["id", "name", "tags", "status", "elapsed_seconds", "note", "input", "output"]
    with open(CSV_RESULTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            row["tags"]   = "|".join(r.get("tags", []))
            row["input"]  = r.get("input", "")[:300]
            row["output"] = r.get("output", "")[:500]
            writer.writerow(row)
    print(f"  CSV  -> {CSV_RESULTS}")


# ─────────────────────────────────────────────────────────────────────────────
#  PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary():
    total = len(results)
    print()
    print("=" * 70)
    print(f"  {'TC_ID':<14} {'Status':<8} {'Time':>7}  Name")
    print(f"  {'-'*14} {'-'*8} {'-'*7}  {'-'*36}")
    for r in results:
        marker = "[PASS]" if r["status"] == "PASS" else "[FAIL]" if r["status"] == "FAIL" else "[ERR ]"
        print(f"  {r['id']:<14} {marker:<8} {r['elapsed_seconds']:>6}s  {r['name'][:50]}")
    print("=" * 70)
    print(f"  TOTAL: {total}   PASS: {pass_count}   FAIL: {fail_count}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global RUN_META

    # write seed doc
    with open(SAMPLE_DOC_PATH, "w", encoding="utf-8") as f:
        f.write(SAMPLE_DOC_TEXT)

    print()
    print("=" * 70)
    print("  Customer Support API - Live Test Runner")
    print(f"  Server : {BASE_URL}")
    print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # check connectivity
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"\n  [OK] Server reachable - HTTP {r.status_code}\n")
    except Exception as e:
        print(f"\n  [FAIL] Cannot reach {BASE_URL}: {e}")
        print("  Is docker-compose up and uvicorn running on port 8000?")
        sys.exit(1)

    session   = requests.Session()
    run_start = time.time()

    # ── Phase 1: Health ─────────────────────────────────────────────────────
    print("Phase 1 - /health")
    print("-" * 40)
    for tc_id, name, fn, tags in STATIC_TESTS:
        if "health" in tags:
            run_test(tc_id, name, fn, session, tags)

    # ── Phase 2: Upload ──────────────────────────────────────────────────────
    print("\nPhase 2 - /v1/documents/upload")
    print("-" * 40)
    seed_index(session)
    for tc_id, name, fn, tags in STATIC_TESTS:
        if "upload" in tags and "reset" not in tags and "e2e" not in tags:
            run_test(tc_id, name, fn, session, tags)

    # ── Phase 3: Query - static ──────────────────────────────────────────────
    print("\nPhase 3 - /v1/query (static)")
    print("-" * 40)
    seed_index(session)   # ensure index is populated
    for tc_id, name, fn, tags in STATIC_TESTS:
        if "query" in tags:
            run_test(tc_id, name, fn, session, tags)

    # ── Phase 4: Query - from test_questions.json ────────────────────────────
    dynamic = load_question_tests()
    if dynamic:
        print(f"\nPhase 4 - /v1/query (dynamic from test_questions.json) [{len(dynamic)} questions]")
        print("-" * 40)
        seed_index(session)
        for tc_id, name, fn, tags in dynamic:
            run_test(tc_id, name, fn, session, tags)

    # ── Phase 5: Reset ───────────────────────────────────────────────────────
    print("\nPhase 5 - /v1/reset")
    print("-" * 40)
    for tc_id, name, fn, tags in STATIC_TESTS:
        if "reset" in tags:
            run_test(tc_id, name, fn, session, tags)

    # ── Phase 6: CORS ────────────────────────────────────────────────────────
    print("\nPhase 6 - CORS")
    print("-" * 40)
    for tc_id, name, fn, tags in STATIC_TESTS:
        if "cors" in tags:
            run_test(tc_id, name, fn, session, tags)

    # ── Phase 7: E2E ─────────────────────────────────────────────────────────
    print("\nPhase 7 - End-to-End")
    print("-" * 40)
    for tc_id, name, fn, tags in STATIC_TESTS:
        if "e2e" in tags:
            run_test(tc_id, name, fn, session, tags)

    # cleanup
    if os.path.exists(SAMPLE_DOC_PATH):
        os.remove(SAMPLE_DOC_PATH)

    # ── update meta ───────────────────────────────────────────────────────────
    RUN_META["run_date"]         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    RUN_META["total"]            = len(results)
    RUN_META["passed"]           = pass_count
    RUN_META["failed"]           = fail_count
    RUN_META["duration_seconds"] = round(time.time() - run_start, 2)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print_summary()
    print("\n  Saving results...")
    save_json()
    save_csv()
    print(f"\n  Done! PASS: {pass_count}/{len(results)}   FAIL: {fail_count}/{len(results)}")


if __name__ == "__main__":
    main()
