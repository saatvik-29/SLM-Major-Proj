"""
Run every question in test_questions.json against the API and print answers.
No pass/fail — just plain Q&A output.

Usage:
    python run_questions.py          # auto-uploads 3 structured SRM data files
    python run_questions.py doc.pdf  # uploads your own document instead
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
import json
import os

BASE_URL       = "http://localhost:8000"
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "test_questions.json")
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")

# The 4 structured data files — uploaded together for best RAG coverage
DEFAULT_DOCS = [
    os.path.join(DATA_DIR, "srm_hostel_policies.txt"),   # overview + all policies
    os.path.join(DATA_DIR, "srm_hostel_girls.txt"),       # all 8 girls blocks
    os.path.join(DATA_DIR, "srm_hostel_boys.txt"),        # all 10 boys blocks
    os.path.join(DATA_DIR, "srm_hostel_wardens.txt"),     # all warden phone numbers + emails
]


def upload_documents(session, doc_paths):
    """Reset the index, then upload all given files in one request."""
    session.post(f"{BASE_URL}/v1/reset")

    files = []
    for path in doc_paths:
        if not os.path.exists(path):
            print(f"  [ERROR] File not found: {path}")
            sys.exit(1)
        fname = os.path.basename(path)
        mime  = "application/pdf" if path.lower().endswith(".pdf") else "text/plain"
        size  = os.path.getsize(path) // 1024
        print(f"  + {fname}  ({size} KB)")
        files.append(("files", (fname, open(path, "rb"), mime)))

    r      = session.post(f"{BASE_URL}/v1/documents/upload", files=files)
    result = r.json()
    chunks = result.get("chunks_indexed", 0)
    print(f"  Total indexed: {chunks} chunks\n")
    if chunks == 0:
        print("  [WARN] 0 chunks indexed — answers will all be 'I don't know'.")


def main():
    # Use explicit arg if given (single file), else upload all 3 default files
    if len(sys.argv) > 1:
        doc_paths = [sys.argv[1]]
    else:
        doc_paths = DEFAULT_DOCS

    # Load questions
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    total = len(questions)

    print(f"\n{'='*70}")
    print(f"  SRM Hostel Q&A — {total} questions")
    print(f"  Server  : {BASE_URL}")
    print(f"{'='*70}\n")

    session = requests.Session()

    print("  Step 1: Seeding index with structured hostel data...")
    upload_documents(session, doc_paths)

    print(f"  Step 2: Running {total} questions...\n")
    print("-" * 70)

    for i, item in enumerate(questions, start=1):
        question = item.get("question", "").strip()
        print(f"[{i}/{total}] Q: {question}")

        try:
            r      = session.post(f"{BASE_URL}/v1/query",
                                  json={"question": question},
                                  timeout=120)
            answer = r.json().get("answer", "(no answer field)")
        except Exception as e:
            answer = f"ERROR: {e}"

        print(f"       A: {answer}")
        print()

    print(f"{'='*70}")
    print(f"  Done — {total} questions answered.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
