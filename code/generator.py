"""
generator.py — TinyLlama GGUF based generator for Sanskrit RAG pipeline.

Model: TinyLlama-1.1B-Chat-v1.0 Q4_K_M (GGUF, ~670MB)
Runtime: ctransformers (CPU only, no GPU needed)

Behavior:
  - English query  → answers in English
  - Sanskrit query → answers in Sanskrit / Devanagari
  - Automatically downloads model on first run
"""

import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LLM_MODEL_PATH,
    LLM_MODEL_URL,
    LLM_TEMPERATURE,
    LLM_CONTEXT_LENGTH,
    LLM_MAX_NEW_TOKENS,
)

# ── Bilingual story context ───────────────────────────────────────────────────
# Injected into every prompt so TinyLlama has English + Sanskrit context
# even when the corpus chunk is pure Devanagari.
STORY_CONTEXT = {
    "मूर्खभृत्यस्य": """Story: The Foolish Servant (मूर्खभृत्यस्य)
Characters: Shankhanaad (शंखनाद) = foolish servant, Govardhanaadas (गोवर्धनदास) = master.
- Asked to bring sugar (शर्करा): carried it in a torn cloth → all spilled on the way.
- Asked to bring a puppy (श्वानशावक): put it in a sealed bag → puppy suffocated and died.
- Asked to bring milk (दुग्ध): tied vessel with rope and dragged it → vessel rolled, milk spilled.
- Told "let your face become black": smeared kajal soot on face → returned with blackened face.
Moral (नीति): वरम् भृत्यविहिनस्य जीवितम् श्रमपूरितम् । मूर्खभृत्यस्य संसर्गात् सर्वम् कार्यम् विनश्यति ॥
(Better to live a hard life without a servant than have a foolish one who ruins everything.)""",

    "चतुरस्य कालीदासस्य": """Story: The Clever Kalidasa (चतुरस्य कालीदासस्य)
Characters: Kalidasa (कालीदास) = clever poet, King Bhoj (भोजराजा) = king.
- King Bhoj announced: any poet reciting a NEW poem in court wins one lakh rupees (लक्षरुप्यकाणि).
- Problem: court scholars memorized every poem after 1, 2, or 3 hearings and claimed to know it.
- So no poet could ever win — scholars always said "we already know this poem."
- Kalidasa's trick: gave a poet a subhashita claiming King Bhoj's father stole 99 crores of gems.
- Scholars couldn't claim to know it (that would confirm the theft) → stayed silent → poet won lakh rupees.
Moral: चतुरः खलु कालीदासः । (Kalidasa was indeed clever.)""",

    "वृद्धायाः चार्तुयम्": """Story: The Old Woman's Cleverness (वृद्धायाः चार्तुयम्)
Characters: Old woman (वृद्धा), King, Monkeys (वानराः), Ghantakarna (घण्टाकर्ण).
- A thief stole a bell (घण्टा), ran into forest, was killed by a tiger. Bell fell in forest.
- Monkeys (वानराः) found the bell and kept ringing it out of curiosity.
- Citizens of Chitrapura heard bell from mountain → feared demon Ghantakarna was eating people.
- Citizens began leaving the city out of fear.
- King announced: whoever kills Ghantakarna gets gold reward.
- Old woman observed forest quietly → discovered it was just monkeys ringing the bell.
- She gave monkeys sweet fruits (मधुराणि फलानि) as distraction → took the bell away.
- Told king "I killed Ghantakarna" → received large amount of gold.
- Citizens freed from fear, returned to city.""",

    "देवभक्तः": """Story: The Devotee and God (देवभक्तः)
Characters: Devotee (भक्त), God (देव), Three helpful strangers.
- Devotee prayed daily: "God give me health and wealth" — but made ZERO effort himself.
- One day travelling by bullock cart on muddy road → heavy rain for 3 hours → cart wheel sank in mud.
- Three different people came one by one and offered help.
- Each time devotee refused: "God will help me, I don't need you."
- Water rose to his neck → he drowned → went to heaven.
- In heaven, devotee asked God: "Why didn't you help me?"
- God said: "I came THREE times as those three helpers. You refused every time. If you make no effort, how can I help?"
Moral: उद्यमः साहसम् धैर्यम् बुद्धिः शक्तिः पराक्रमः । षडेते यत्र वर्तन्ते तत्र देवः साहाय्यकृत् ॥
(Effort, courage, patience, wisdom, strength, valor — where these six exist, God helps.)""",

    "शीतं बहु बाधति": """Story: The Cold Hurts (शीतं बहु बाधति)
Characters: Kalidasa (कालीदास) disguised as palanquin carrier, Foreign Scholar (परदेशीय पण्डित).
- Foreign scholar sent message to King Bhoj: "I will come on X date to debate your court scholars."
- Kalidasa disguised himself as a palanquin carrier (पालखीधारक) to receive the scholar.
- Scholar did not recognize Kalidasa.
- It was winter (शिशिर). Scholar said: "शीतं बहु बाधति" (The cold hurts very much).
- Grammar mistake: verb "badh" (बाध्) is Atmanepadi (आत्मनेपदी) → correct form is "बाधते" not "बाधति".
- Kalidasa immediately corrected him cleverly.
- Scholar thought: "If even palanquin carriers here know Sanskrit grammar this well, the court scholars will surely defeat me."
- Scholar ordered Kalidasa to turn back → went home without entering the court.""",
}

SECTION_MAP = [
    ("मूर्खभृत्यस्य",    "मूर्खभृत्यस्य"),
    ("कालीदासस्य",       "चतुरस्य कालीदासस्य"),
    ("चतुरस्य",          "चतुरस्य कालीदासस्य"),
    ("वृद्धायाः",         "वृद्धायाः चार्तुयम्"),
    ("घण्टाकर्ण",         "वृद्धायाः चार्तुयम्"),
    ("देवभक्तः",          "देवभक्तः"),
    ("परमः देवभक्तः",     "देवभक्तः"),
    ("शीतं",             "शीतं बहु बाधति"),
    ("सर्वे जानन्ति",     "शीतं बहु बाधति"),
]


class SanskritGenerator:

    def __init__(self):
        self._llm     = None
        self._backend = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load TinyLlama GGUF. Downloads automatically if not present."""
        if self._try_load_gguf():
            return
        print("[Generator] GGUF failed — falling back to knowledge-base mode.")
        self._backend = "knowledge_base"

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        if self._backend is None:
            raise RuntimeError("Generator not loaded. Call load() first.")

        is_sanskrit = self._has_devanagari(query)
        context     = self._build_context(context_chunks)

        t0 = time.time()

        if self._backend == "gguf":
            answer = self._run_gguf(query, context, is_sanskrit)
        else:
            answer = self._knowledge_base_answer(query, context_chunks, is_sanskrit)

        print(f"[Generator] Answer in {'Sanskrit' if is_sanskrit else 'English'} "
              f"in {time.time()-t0:.1f}s  [{self._backend}]")
        return answer.strip()

    # ── GGUF inference ────────────────────────────────────────────────────────

    def _try_load_gguf(self) -> bool:
        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError:
            print("[Generator] ctransformers not installed.")
            print("[Generator] Run: pip install ctransformers")
            return False

        model_path = LLM_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"[Generator] GGUF model not found at '{model_path}'.")
            print("[Generator] Downloading TinyLlama-1.1B-Chat GGUF (~670 MB)...")
            if not self._download_model(model_path):
                return False

        try:
            print(f"[Generator] Loading TinyLlama GGUF from '{model_path}' ...")
            from ctransformers import AutoModelForCausalLM
            self._llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="llama",
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                context_length=LLM_CONTEXT_LENGTH,
                threads=os.cpu_count(),
            )
            self._backend = "gguf"
            print("[Generator] TinyLlama GGUF loaded ✓")
            return True
        except Exception as e:
            print(f"[Generator] GGUF load error: {e}")
            return False

    def _run_gguf(self, query: str, context: str, is_sanskrit: bool) -> str:
        """Build prompt and run TinyLlama inference."""
        if is_sanskrit:
            instruction = (
                "You are an expert in Sanskrit literature. "
                "The user asked a question in Sanskrit (Devanagari). "
                "Read the context carefully and answer in Sanskrit (Devanagari script). "
                "Be specific and accurate."
            )
        else:
            instruction = (
                "You are an expert in Sanskrit literature. "
                "Read the context carefully and answer the question in English. "
                "Be specific, concise and accurate. "
                "Do not make up information not present in the context."
            )

        prompt = (
            f"<|system|>\n{instruction}\n</s>\n"
            f"<|user|>\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"</s>\n"
            f"<|assistant|>\n"
        )

        raw = self._llm(prompt)

        # Clean up — remove any prompt echo
        if "<|assistant|>" in raw:
            raw = raw.split("<|assistant|>")[-1]
        return raw.strip()

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context by combining:
        1. Bilingual story summary (English + Sanskrit) for the matched story
        2. Actual retrieved chunk text
        """
        parts     = []
        seen_keys = set()

        for chunk in chunks:
            section   = chunk.get("metadata", {}).get("section", "")
            story_key = self._section_to_story_key(section)

            if story_key and story_key not in seen_keys:
                parts.append(STORY_CONTEXT[story_key])
                seen_keys.add(story_key)

        # Also add raw chunk text (may have partial Devanagari / English)
        chunk_texts = []
        for chunk in chunks[:2]:  # top 2 chunks only
            text = chunk.get("text", "").strip()
            if text:
                chunk_texts.append(text)

        if chunk_texts:
            parts.append("--- Retrieved text from corpus ---")
            parts.extend(chunk_texts)

        return "\n\n".join(parts)

    def _section_to_story_key(self, section: str) -> Optional[str]:
        for fragment, key in SECTION_MAP:
            if fragment in section:
                return key
        return None

    # ── Knowledge base fallback ───────────────────────────────────────────────

    def _knowledge_base_answer(
        self, query: str, chunks: List[Dict], is_sanskrit: bool
    ) -> str:
        """Simple keyword-based fallback when GGUF is unavailable."""
        story_key = None
        for chunk in chunks:
            section   = chunk.get("metadata", {}).get("section", "")
            story_key = self._section_to_story_key(section)
            if story_key:
                break

        if not story_key:
            return (
                "संदर्भे उत्तरं न प्राप्तम् ।"
                if is_sanskrit
                else "Answer not found in context."
            )

        # Return story context summary
        ctx = STORY_CONTEXT[story_key]
        if is_sanskrit:
            # Return Devanagari lines only
            lines = [l for l in ctx.splitlines() if self._has_devanagari(l)]
            return "\n".join(lines) if lines else ctx
        else:
            # Return English lines only
            lines = [l for l in ctx.splitlines() if l and not self._has_devanagari(l)]
            return "\n".join(lines) if lines else ctx

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _has_devanagari(text: str) -> bool:
        return bool(re.search(r"[\u0900-\u097F]", text))

    @staticmethod
    def _download_model(dest_path: str) -> bool:
        import urllib.request
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            def _progress(count, block, total):
                pct = min(count * block * 100 // total, 100)
                print(f"\r  Downloading... {pct}%", end="", flush=True)

            urllib.request.urlretrieve(LLM_MODEL_URL, dest_path, reporthook=_progress)
            print(f"\n[Generator] Model saved → {dest_path}")
            return True
        except Exception as e:
            print(f"\n[Generator] Download failed: {e}")
            print(f"[Generator] Manually download from:\n  {LLM_MODEL_URL}")
            print(f"[Generator] Place at: {dest_path}")
            return False