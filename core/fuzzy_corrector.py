"""
模組 3：SymSpell 英文 + 香港商業實體模糊校正
- 用 importlib.resources 取字典路徑（pkg_resources 已被棄用）
- 載入失敗唔會拖死成個服務，只係停用呢個模組
"""
import logging
from importlib.resources import files
from symspellpy import SymSpell, Verbosity

logger = logging.getLogger("DocuOCR.Fuzzy")


class SymSpellCorrector:
    def __init__(self):
        self.sym_spell = None
        try:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dict_path = files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            self.sym_spell.load_dictionary(str(dict_path), term_index=0, count_index=1)
            self._load_hk_business_lexicon()
            logger.info("✅ SymSpell 字典載入完成")
        except Exception as e:
            logger.warning(f"SymSpell 載入失敗，英文模糊校正已停用：{e}")

    def _load_hk_business_lexicon(self):
        common_terms = [
            "limited", "company", "corporation", "kowloon", "hongkong", "hong",
            "kong", "central", "express", "invoice", "receipt", "signature",
            "payment", "account", "reference", "cheque", "address", "shipping",
            "statement", "balance", "amount", "total", "business", "registration",
            "secretary", "director", "shareholder",
        ]
        for term in common_terms:
            self.sym_spell.create_dictionary_entry(term, 100000)

    def correct_line(self, text: str) -> str:
        if not self.sym_spell or not text:
            return text
        words = text.split(" ")
        result = []
        for word in words:
            # 淨係校全小寫、長度足夠嘅英文單詞；全大寫（縮寫）唔掂
            if word.isalpha() and len(word) >= 4 and not word.isupper():
                suggestions = self.sym_spell.lookup(
                    word.lower(), Verbosity.TOP, max_edit_distance=1
                )
                if suggestions and suggestions[0].count > 500:
                    result.append(suggestions[0].term)
                    continue
            result.append(word)
        return " ".join(result)
