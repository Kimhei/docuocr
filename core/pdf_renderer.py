"""
雙層可搜尋 PDF 渲染器
- 用 Noto Sans CJK 嵌入文字層（支援繁體 + 廣東話 HKSCS 字元）
- render_mode=3：隱形文字（淨係用嚟搜尋 / 複製，唔遮原圖）
- 每頁處理完即 malloc_trim，將記憶體還返畀 OS
"""
import os
import gc
import ctypes
import logging

import fitz

logger = logging.getLogger("DocuOCR.PDF")

NOTO_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


class PDFSearchableRenderer:
    @staticmethod
    def force_malloc_trim():
        gc.collect()
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass

    @classmethod
    def render_searchable_pdf(
        cls,
        doc: fitz.Document,
        page_idx: int,
        ocr_results: list,
        scale_x: float,
        scale_y: float,
    ):
        page = doc[page_idx]
        use_noto = os.path.exists(NOTO_FONT_PATH)
        if not use_noto:
            logger.warning("搵唔到 Noto Sans CJK，改用內建 china-t 字體")

        for line in ocr_results:
            box, text, _score = line
            if not text or not text.strip():
                continue
            x0 = min(pt[0] for pt in box) * scale_x
            y0 = min(pt[1] for pt in box) * scale_y
            x1 = max(pt[0] for pt in box) * scale_x
            y1 = max(pt[1] for pt in box) * scale_y

            rect = fitz.Rect(x0, y0, x1, y1)
            if rect.width <= 0 or rect.height <= 0:
                continue
            font_size = max(int(rect.height * 0.75), 6)

            try:
                if use_noto:
                    page.insert_textbox(
                        rect, text,
                        fontfile=NOTO_FONT_PATH,
                        fontsize=font_size,
                        render_mode=3,  # 隱形文字層
                    )
                else:
                    page.insert_textbox(
                        rect, text,
                        fontname="china-t",
                        fontsize=font_size,
                        render_mode=3,
                    )
            except Exception:
                pass

        cls.force_malloc_trim()
