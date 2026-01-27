import json
import re
from pathlib import Path
from typing import List, Dict, Set, Optional
from fastapi import FastAPI, HTTPException

from src.path import OUTPUT_JSON_DIR, OUTPUT_DIR

# =========================================================
# Configuration
# =========================================================

SCOPE_DIR = OUTPUT_DIR / "scope"
SCOPE_DIR.mkdir(parents=True, exist_ok=True)

MIN_SUMMARY_WORDS = 60
MAX_SUMMARY_WORDS = 80

# =========================================================
# Regex Patterns
# =========================================================

SCOPE_HEADER_RE = re.compile(r">\s*(\d+\.?\s*)?(Scope|SCOPE)\s*<")
CLAUSE_1_RE = re.compile(r">\s*1(\.|\s|<)")
HTML_TAG_RE = re.compile(r"<[^>]+>")

TEST_SECTION_RE = re.compile(
    r"^\s*(\d+\.)*\d+\s+.*\btest(s|ing)?\b",
    re.IGNORECASE
)

# =========================================================
# Utilities
# =========================================================

def clean_html(html: str) -> str:
    if not html:
        return ""
    text = HTML_TAG_RE.sub("", html)
    return " ".join(text.split()).strip()

def is_english(text: str, threshold: float = 0.80) -> bool:
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / len(text) > threshold

def starts_with_section_number(text: str) -> bool:
    return bool(re.match(r'^\d+(\.\d+)*\s+', text))

# =========================================================
# NEW: Scope Summary (ADDED ONLY)
# =========================================================

def build_scope_summary(scope_lines: List[str]) -> str:
    """
    Build a 40–60 word extractive summary from scope text.
    Uses ONLY original scope text. No paraphrasing.
    """
    if not scope_lines:
        return ""

    full_text = " ".join(s.strip() for s in scope_lines if s.strip())
    words = full_text.split()

    if len(words) <= MAX_SUMMARY_WORDS:
        return full_text.strip()

    return " ".join(words[:MAX_SUMMARY_WORDS]).strip()

# =========================================================
# Title Extraction (UNCHANGED)
# =========================================================

def extract_document_title(blocks: List[Dict]) -> Optional[str]:
    for block in blocks:
        if block.get("block_type") != "SectionHeader":
            continue
        text = clean_html(block.get("html", ""))
        if text and is_english(text) and not starts_with_section_number(text):
            return text
    return None

# =========================================================
# Scope Extraction (UNCHANGED)
# =========================================================

def extract_scope(json_path: Path) -> List[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    scope_text = []
    collecting = False
    scope_hierarchy = None
    clause1_candidate = []

    for page in data.get("children", []):
        if page.get("block_type") != "Page":
            continue

        for block in page.get("children", []):
            html = block.get("html", "")
            hierarchy = block.get("section_hierarchy")

            if SCOPE_HEADER_RE.search(html):
                collecting = True
                scope_hierarchy = hierarchy
                continue

            if collecting and block.get("block_type") == "SectionHeader":
                if hierarchy != scope_hierarchy:
                    collecting = False

            if collecting and block.get("block_type") == "Text":
                txt = clean_html(html)
                if txt:
                    scope_text.append(txt)

            if not scope_text and CLAUSE_1_RE.search(html):
                collecting = "clause1"
                continue

            if collecting == "clause1" and block.get("block_type") == "Text":
                txt = clean_html(html)
                if txt:
                    clause1_candidate.append(txt)

            if collecting == "clause1" and block.get("block_type") == "SectionHeader":
                collecting = False

    return scope_text or clause1_candidate

# =========================================================
# Test Extraction (UNCHANGED)
# =========================================================

def extract_test_sections(blocks: List[Dict]) -> List[str]:
    tests = []
    seen: Set[str] = set()

    for block in blocks:
        if block.get("block_type") != "SectionHeader":
            continue
        text = clean_html(block.get("html", ""))
        if TEST_SECTION_RE.match(text) and text not in seen:
            tests.append(text)
            seen.add(text)

    return tests

# =========================================================
# Document Processing (ONLY summary added)
# =========================================================

def process_document(json_path: Path) -> Dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    blocks = []
    for page in data.get("children", []):
        if page.get("block_type") == "Page":
            blocks.extend(page.get("children", []))

    scope = extract_scope(json_path)

    return {
        "document_id": json_path.stem,
        "document_title": extract_document_title(blocks),
        "scope": scope,
        "summary": build_scope_summary(scope),   # ✅ ADDED
        "tests": extract_test_sections(blocks),
    }

def save_document(output: Dict):
    out_path = SCOPE_DIR / f"{output['document_id']}_scope.json"
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# =========================================================
# FastAPI
# =========================================================

app = FastAPI(title="Standards Extraction API")

@app.post("/scope/extract")
def extract_scope_api():
    json_files = list(OUTPUT_JSON_DIR.glob("*.json"))
    if not json_files:
        raise HTTPException(404, "No JSON files found")

    processed = []

    for json_file in json_files:
        output = process_document(json_file)
        if output["scope"] or output["tests"]:
            save_document(output)
            processed.append(output["document_id"])

    return {
        "status": "success",
        "documents_processed": len(processed),
        "processed_documents": processed,
        "output_dir": str(SCOPE_DIR),
    }