"""
document_loader.py — Load and preprocess Sanskrit documents into text chunks.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import CORPUS_FILE, CHUNK_SIZE, CHUNK_OVERLAP, CHUNKS_FILE


def make_chunk(text: str, chunk_id: int, source: str, metadata: dict = None) -> Dict:
    return {
        "id":         chunk_id,
        "text":       text.strip(),
        "source":     source,
        "metadata":   metadata or {},
        "char_count": len(text.strip()),
    }


class SanskritDocumentLoader:

    KNOWN_TITLES = [
        "मूर्खभृत्यस्य",
        "चतुरस्य कालीदासस्य",
        "वृद्धायाः चार्तुयम्",
        "शीतं बहु बाधति",
        "एकः परमः देवभक्तः",
        "देवभक्तः",
    ]

    def __init__(
        self,
        corpus_path: str = str(CORPUS_FILE),
        chunk_size:  int = CHUNK_SIZE,
        overlap:     int = CHUNK_OVERLAP,
    ):
        self.corpus_path = Path(corpus_path)
        self.chunk_size  = chunk_size
        self.overlap     = overlap

    def load(self) -> List[Dict]:
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")

        raw_text = self.corpus_path.read_text(encoding="utf-8")
        raw_text = self._clean_text(raw_text)
        sections = self._split_into_sections(raw_text)
        print(f"[Loader] Found {len(sections)} section(s) in corpus.")

        chunks: List[Dict] = []
        chunk_id = 0

        for section_title, section_text in tqdm(sections, desc="Chunking sections"):
            for chunk_text in self._sliding_window_chunks(section_text):
                if len(chunk_text.strip()) < 30:
                    continue
                chunks.append(make_chunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source=self.corpus_path.name,
                    metadata={"section": section_title},
                ))
                chunk_id += 1

        print(f"[Loader] Created {len(chunks)} chunk(s) total.")
        return chunks

    def save_chunks(self, chunks: List[Dict], path: str = CHUNKS_FILE) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Loader] Chunks saved → {path}")

    def load_chunks(self, path: str = CHUNKS_FILE) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_into_sections(self, text: str) -> List[tuple]:
        # Try markdown headings first
        sections = self._split_by_markdown(text)
        if len(sections) > 1:
            print("[Loader] Split strategy: markdown headings")
            return sections
        # Try known titles
        sections = self._split_by_known_titles(text)
        if len(sections) > 1:
            print("[Loader] Split strategy: known story titles")
            return sections
        print("[Loader] Split strategy: single section")
        return [("Full Corpus", text)]

    def _split_by_markdown(self, text: str) -> List[tuple]:
        parts    = re.split(r"^## (.+)$", text, flags=re.MULTILINE)
        sections = []
        if parts[0].strip():
            sections.append(("Introduction", parts[0].strip()))
        for i in range(1, len(parts) - 1, 2):
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if body:
                sections.append((parts[i].strip(), body))
        return sections

    def _split_by_known_titles(self, text: str) -> List[tuple]:
        lines           = text.split("\n")
        sections        = []
        current_title   = "Introduction"
        current_lines   = []

        for line in lines:
            stripped = line.strip()
            if self._is_title_line(stripped):
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_title, body))
                current_title = stripped
                current_lines = []
            else:
                current_lines.append(line)

        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, body))
        return sections if len(sections) > 1 else []

    def _is_title_line(self, line: str) -> bool:
        if not line:
            return False
        for title in self.KNOWN_TITLES:
            if title in line:
                return True
        return False

    def _sliding_window_chunks(self, text: str) -> List[str]:
        chunks   = []
        start    = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                end = self._find_boundary(text, end)
            chunks.append(text[start:end])
            start += max(1, self.chunk_size - self.overlap)
        return chunks

    def _find_boundary(self, text: str, pos: int, window: int = 80) -> int:
        search_end = min(pos + window, len(text))
        for sep in ("।।", "।", "\n\n", ". ", "! ", "? "):
            idx = text.find(sep, pos, search_end)
            if idx != -1:
                return idx + len(sep)
        return pos