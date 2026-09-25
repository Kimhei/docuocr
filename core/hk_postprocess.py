"""
模組 1：OpenCC（s2hk）+ 動態反饋字典熱重載
- 用 opencc-python-reimplemented（純 Python，ARM 免編譯）
- opencc 缺席都唔會死，自動跳過簡轉港繁步驟
- 用 threading.Lock 保護字典熱更新，避免同 OCR 線程打架
"""
import os
import re
import json
import threading
import logging

logger = logging.getLogger("DocuOCR.Lexicon")

try:
    from opencc import OpenCC
    _HAS_OPENCC = True
except ImportError:
    OpenCC = None
    _HAS_OPENCC = False

# 廣東話常用單字白名單（避免全域誤殺）
HK_SINGLE_CHAR_ALLOWLIST = "喺佢啲冇咗諗𨋢嘢呎"


class HKPostProcessor:
    def __init__(self, lexicon_dir: str = "./lexicons"):
        self._lock = threading.Lock()
        self.lexicon_dir = lexicon_dir
        self.feedback_file = os.path.join(self.lexicon_dir, "user_feedback.json")
        self.converter = OpenCC("s2hk") if _HAS_OPENCC else None
        if not _HAS_OPENCC:
            logger.warning("opencc 未安裝，已跳過 s2hk 轉換（只用自訂字典）")
        self.dictionary: dict = {}
        self.compiled_regex = None
        self.load_lexicons()

    def load_lexicons(self):
        combined: dict = {}
        os.makedirs(self.lexicon_dir, exist_ok=True)
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, "w", encoding="utf-8") as f:
                json.dump({}, f)

        for file_name in sorted(os.listdir(self.lexicon_dir)):
            if not file_name.endswith(".json"):
                continue
            fp = os.path.join(self.lexicon_dir, file_name)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if len(k) >= 2 or (len(k) == 1 and k in HK_SINGLE_CHAR_ALLOWLIST):
                        combined[k] = v
            except Exception as e:
                logger.warning(f"載入字典失敗 {file_name}: {e}")

        with self._lock:
            self.dictionary = combined
            self._recompile_locked()
        logger.info(f"✅ 字典載入完成，共 {len(combined)} 條規則")

    def _recompile_locked(self):
        if self.dictionary:
            sorted_keys = sorted(self.dictionary.keys(), key=len, reverse=True)
            pattern = "|".join(re.escape(k) for k in sorted_keys)
            self.compiled_regex = re.compile(pattern)
        else:
            self.compiled_regex = None

    def add_feedback_entry(self, wrong: str, correct: str):
        """使用者在線修正 → 寫入 JSON → 即時熱更新正則樹"""
        wrong, correct = wrong.strip(), correct.strip()
        data: dict = {}
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data[wrong] = correct
        tmp = self.feedback_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.feedback_file)  # 原子寫入，唔怕中途斷電爛檔

        with self._lock:
            self.dictionary[wrong] = correct
            self._recompile_locked()

    def process(self, text: str) -> str:
        if not text:
            return ""
        if self.converter:
            try:
                text = self.converter.convert(text)
            except Exception:
                pass
        with self._lock:
            rx, d = self.compiled_regex, self.dictionary
        if rx:
            text = rx.sub(lambda m: d[m.group(0)], text)
        return text
