"""
DocuOCR Enterprise Pro — ARM64 + 2GB RAM 特化版主服務

相對舊版嘅實質改動：
  1. OCR 雙引擎改為「懶加載」：idle 時唔佔 RAM，第一次請求先載入（約 10–30 秒）。
  2. 預設併發 = 1（環境變數 MAX_CONCURRENT_JOBS 可調，2GB 機唔建議調大）。
  3. PDF 渲染唔經 PNG encode/decode roundtrip，直接用 pixmap samples → numpy。
  4. DPI 自動上限：頁面像素邊長超過 MAX_PAGE_DIM（預設 3500px）會自動降 DPI。
  5. 新增 /api/v1/ocr/image：單張圖片 OCR，回傳 JSON。
  6. SLM 缺席時自動降級，唔會擲錯。
"""
import os
import gc
import fitz
import asyncio
import logging
import platform
import tempfile
import subprocess

import cv2
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from core.ocr_engine import HandwritingDualEngine, SLM_PATH
from core.pdf_renderer import PDFSearchableRenderer
from core.hk_postprocess import HKPostProcessor, HK_SINGLE_CHAR_ALLOWLIST
from core.fuzzy_corrector import SymSpellCorrector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DocuOCR")

app = FastAPI(title="DocuOCR Enterprise Pro", version="5.0-arm64-2gb")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- 環境參數 ----------------
MAX_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
ENABLE_SLM = os.environ.get("ENABLE_SLM", "0") == "1"
SLM_TIMEOUT = int(os.environ.get("SLM_TIMEOUT", "30"))
MAX_PAGE_DIM = int(os.environ.get("MAX_PAGE_DIM", "3500"))
NATIVE_TEXT_MIN_CHARS = int(os.environ.get("NATIVE_TEXT_MIN_CHARS", "50"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))

# ---------------- 輕量組件：啟動即載 ----------------
hk_processor = HKPostProcessor()
fuzzy_corrector = SymSpellCorrector()

# ---------------- 重量組件：懶加載 ----------------
_ocr_engine: HandwritingDualEngine | None = None
_engine_lock = asyncio.Lock()
processing_semaphore = asyncio.Semaphore(MAX_JOBS)


async def get_engine() -> HandwritingDualEngine:
    global _ocr_engine
    if _ocr_engine is None:
        async with _engine_lock:
            if _ocr_engine is None:
                logger.info("🔧 首次請求：載入 OCR 雙引擎（約需 10–30 秒）…")
                _ocr_engine = await asyncio.to_thread(HandwritingDualEngine)
                logger.info("✅ 引擎就緒")
    return _ocr_engine


@app.on_event("startup")
async def _maybe_prewarm_engine():
    # PREWARM_ON_STARTUP=1：開機後喺背景預載模型，第一次 request 唔使等；
    # 代價係開機頭 1 分鐘 CPU 較忙。預設 0（保持開機快）。
    if os.environ.get("PREWARM_ON_STARTUP", "0") == "1":
        async def _warm():
            try:
                await get_engine()
                logger.info("🔥 引擎背景預熱完成")
            except Exception as e:
                logger.warning(f"背景預熱失敗（第一次請求時會再試）：{e}")
        asyncio.create_task(_warm())
        logger.info("🔥 已排程背景預熱…")


def slm_model_ready() -> bool:
    return os.path.exists(SLM_PATH) and os.path.getsize(SLM_PATH) > 0


# ---------------- 工具函數 ----------------
class FeedbackModel(BaseModel):
    wrong_text: str = Field(..., min_length=1, max_length=50)
    correct_text: str = Field(..., min_length=1, max_length=50)


def cleanup_files(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


def is_native_text_page(page: fitz.Page) -> bool:
    clean_text = "".join(page.get_text("text").split())
    return len(clean_text) >= NATIVE_TEXT_MIN_CHARS


def parse_page_range(range_str: str, total_pages: int) -> list:
    if not range_str:
        return list(range(total_pages))
    pages = set()
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            try:
                s, e = part.split("-")
                pages.update(range(max(0, int(s) - 1), min(total_pages, int(e))))
            except Exception:
                pass
        elif part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                pages.add(idx)
    return sorted(pages) if pages else list(range(total_pages))


def pixmap_to_bgr(page: fitz.Page, dpi: int) -> tuple[np.ndarray, int, int]:
    """將 PDF 頁轉做 BGR numpy；太大頁自動降 DPI。唔經 PNG roundtrip，慳 CPU 同 RAM。"""
    eff_dpi = dpi
    pix = page.get_pixmap(dpi=eff_dpi, alpha=False)
    while max(pix.width, pix.height) > MAX_PAGE_DIM and eff_dpi > 72:
        eff_dpi = int(eff_dpi * 0.8)
        pix = page.get_pixmap(dpi=eff_dpi, alpha=False)

    w, h = pix.width, pix.height
    if pix.n == 3:  # RGB，直接 reshape
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(h, w, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    else:  # 罕見情況：fallback 經 PNG
        img = cv2.imdecode(
            np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR
        )
    pix = None
    return img, w, h


def clean_ocr_lines(raw_results: list) -> list:
    """模組 1+3：OpenCC s2hk → 香港字典 → SymSpell"""
    cleaned = []
    for box, text, score in raw_results:
        t = fuzzy_corrector.correct_line(hk_processor.process(text))
        cleaned.append((box, t, score))
    return cleaned


def run_slm_subprocess(raw_text: str) -> str:
    """模組 4：獨立子進程跑 Qwen2.5-0.5B，完成即退出，記憶體 100% 釋放"""
    try:
        proc = subprocess.Popen(
            ["python3", "-m", "core.slm_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, _ = proc.communicate(input=raw_text, timeout=SLM_TIMEOUT)
        return stdout.strip() if stdout else raw_text
    except Exception as e:
        logger.warning(f"SLM 子進程逾時或出錯，已跳過：{e}")
        return raw_text


def maybe_slm_smooth(lines: list, avg_score: float, enable_slm: bool, page_no: int) -> list:
    need = (enable_slm or avg_score < 0.75) and slm_model_ready()
    if not need or not lines:
        return lines
    logger.info(f"🧠 [第 {page_no} 頁] 喚醒 Qwen2.5-0.5B 進行語境平滑…")
    page_raw = "\n".join(line[1] for line in lines)
    smoothed = run_slm_subprocess(page_raw).split("\n")
    if len(smoothed) == len(lines):
        return [(lines[i][0], smoothed[i], lines[i][2]) for i in range(len(lines))]
    logger.warning("SLM 回傳行數不符，已保留原文")
    return lines


def run_pdf_pipeline(
    engine: HandwritingDualEngine,
    in_path: str,
    out_path: str,
    dpi: int,
    page_range: str,
    threshold: float,
    skip_native: bool,
    enable_slm: bool,
):
    doc = fitz.open(in_path)
    try:
        total_pages = len(doc)
        target_pages = parse_page_range(page_range, total_pages)
        logger.info(f"📄 共 {total_pages} 頁，處理 {len(target_pages)} 頁（dpi={dpi}）")

        for page_idx in target_pages:
            page = doc[page_idx]

            if skip_native and is_native_text_page(page):
                logger.info(f"⏩ [第 {page_idx + 1} 頁] 原生向量文字，跳過 OCR")
                continue

            img, pw, ph = pixmap_to_bgr(page, dpi)
            raw_results, avg_score = engine.process_image(img, threshold)
            img = None

            if raw_results:
                lines = clean_ocr_lines(raw_results)
                lines = maybe_slm_smooth(lines, avg_score, enable_slm, page_idx + 1)
                scale_x = page.rect.width / pw
                scale_y = page.rect.height / ph
                PDFSearchableRenderer.render_searchable_pdf(
                    doc, page_idx, lines, scale_x, scale_y
                )

            gc.collect()
            PDFSearchableRenderer.force_malloc_trim()

        doc.save(out_path, garbage=4, deflate=True)
    finally:
        doc.close()
        PDFSearchableRenderer.force_malloc_trim()


# ---------------- API ----------------
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    idx = os.path.join("static", "index.html")
    if os.path.exists(idx):
        with open(idx, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>DocuOCR Enterprise Pro</h1><p>static/index.html 缺失</p>"


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "arch": platform.machine(),
        "engine_loaded": _ocr_engine is not None,
        "slm_ready": slm_model_ready(),
        "max_concurrent_jobs": MAX_JOBS,
        "mode": "arm64-2gb-tuned",
    }


@app.post("/api/v1/feedback")
async def save_user_feedback(feedback: FeedbackModel):
    """模組 1：人類校正反饋 API"""
    wrong = feedback.wrong_text.strip()
    correct = feedback.correct_text.strip()
    if wrong == correct:
        raise HTTPException(status_code=400, detail="錯字與正字不能相同。")
    if len(wrong) < 2 and wrong not in HK_SINGLE_CHAR_ALLOWLIST:
        raise HTTPException(
            status_code=400, detail="為防止全域誤殺，替換長度需 ≥ 2 字元（廣東話常用字除外）。"
        )
    try:
        hk_processor.add_feedback_entry(wrong, correct)
        return {"status": "success", "message": f"成功學習：'{wrong}' → '{correct}'（即刻生效）"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ocr/pdf")
async def process_pdf_api(
    file: UploadFile = File(...),
    dpi: int = Form(200),
    page_range: str = Form(""),
    handwriting_threshold: float = Form(0.85),
    skip_native_text: bool = Form(True),
    enable_slm: bool = Form(False),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支援 PDF 檔案格式。")
    dpi = max(72, min(int(dpi), 600))

    async with processing_semaphore:
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"檔案太大（上限 {MAX_UPLOAD_MB}MB）。")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
            tmp_in.write(data)
            in_path = tmp_in.name
        out_path = in_path.replace(".pdf", "_searchable.pdf")

        try:
            engine = await get_engine()
            await asyncio.to_thread(
                run_pdf_pipeline, engine, in_path, out_path, dpi, page_range,
                handwriting_threshold, skip_native_text, enable_slm,
            )
            return FileResponse(
                path=out_path,
                filename=f"Searchable_{file.filename}",
                media_type="application/pdf",
                background=BackgroundTask(cleanup_files, in_path, out_path),
            )
        except Exception as e:
            cleanup_files(in_path, out_path)
            logger.error(f"OCR 流程異常: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ocr/image")
async def process_image_api(
    file: UploadFile = File(...),
    handwriting_threshold: float = Form(0.85),
    enable_slm: bool = Form(False),
):
    """單張圖片 OCR，回傳 JSON（含每行文字、座標、信心分數）"""
    ext = (file.filename or "").lower().rsplit(".", 1)
    if len(ext) < 2 or ext[1] not in ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"):
        raise HTTPException(status_code=400, detail="只支援 png/jpg/webp/bmp/tif 圖片。")

    data = await file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="圖片太大（上限 25MB）。")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="圖片解碼失敗。")

    async with processing_semaphore:
        try:
            engine = await get_engine()
            raw_results, avg_score = await asyncio.to_thread(
                engine.process_image, img, float(handwriting_threshold)
            )
            lines = clean_ocr_lines(raw_results)
            lines = maybe_slm_smooth(lines, avg_score, bool(enable_slm), 1)
            return JSONResponse(
                {
                    "avg_score": round(avg_score, 4),
                    "line_count": len(lines),
                    "lines": [
                        {"text": t, "score": round(float(s), 4), "box": b}
                        for b, t, s in lines
                    ],
                }
            )
        except Exception as e:
            logger.error(f"圖片 OCR 異常: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            gc.collect()
