"""
PP-OCRv5 雙通道 OCR — ARM64 + 2GB RAM 特化，手寫中英文強化版

模型（RapidAI 官方 ModelScope 鏡像，已驗證可下載）：
  det: ch_PP-OCRv5_server_det.onnx
  rec: ch_PP-OCRv5_rec_server_infer.onnx

點解由 v6/v3 轉做 PP-OCRv5（PaddleOCR 3.0）：
  1. 官方明確「大幅改善手寫識別」——複雜連筆、非規範手寫，相對 v4 提升約 13 個百分點。
  2. 單一 rec 模型同時覆蓋簡/繁中文、拼音、英文、日文；香港文件成日中英夾雜，唔使切模型。
  3. 同 rapidocr_onnxruntime 完全兼容：CTC 解碼、[3,48,320] 輸入、(x/255-0.5)/0.5 正規化，
     字符表內嵌喺 ONNX metadata（自動讀取，唔使字典檔）。
  4. 舊嘅 PP-OCRv3 手寫模型（2021 年）已退役：原下載連結失效，精度亦全面落後 v5。

流程：
  1. 全頁 det + rec（v5 server）。
  2. 低分行 → 透視校正裁片 → 細裁片預放大 → CLAHE 對比增強 → v5 rec-only 再認一次，
     同原結果打擂台揀高分者。
     （手寫行通常歪斜、細隻、淺色；校正過嘅輸入質素高好多，rec-only 成本亦好低。）

相對舊版嘅實質修正（已用源碼驗證過 rapidocr_onnxruntime 參數系統）：
  1. 舊版 `rm_session_options=...` 係無效參數，傳入去完全冇作用；
     正確做法係 `intra_op_num_threads` / `inter_op_num_threads`（會自動 propagate 去 Det/Rec）。
  2. 舊版手寫引擎 `det_use=False` 寫錯咗（正確係 `use_det=False`），舊寫法根本冇關到 det；
     而家 rec-only 引擎直接用 TextRecognizer，從源頭就冇 det。
  3. `use_cls=False`：唔做文字方向分類，慳起成個 cls ONNX 模型嘅 RAM。
  4. SLM GGUF 預設唔下載（慳 400MB），ENABLE_SLM=1 先下載。
"""
import os
import time
import logging
import urllib.request
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer
from core.adaptive_image import AdaptiveImageTuner

logger = logging.getLogger("DocuOCR.Engine")

MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

# RapidAI 官方 ModelScope 鏡像（已驗證可下載）
# OCR_PROFILE=quality（預設）：v5 server 模型，準確率最高
# OCR_PROFILE=fast：v5 mobile 模型（det ~5MB / rec ~17MB），檢測快幾倍，RAM 更低，
#   代價係手寫/細字準確率下降；純印刷體文件先好用。
_QUALITY_DET_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/det/ch_PP-OCRv5_server_det.onnx"
_QUALITY_REC_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_server_infer.onnx"
_FAST_DET_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.1/onnx/PP-OCRv5/det/ch_PP-OCRv5_det_mobile.onnx"
_FAST_REC_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.1/onnx/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile.onnx"

OCR_PROFILE = os.environ.get("OCR_PROFILE", "quality").strip().lower()
if OCR_PROFILE == "fast":
    DET_MODEL_URL, REC_MODEL_URL = _FAST_DET_URL, _FAST_REC_URL
else:
    if OCR_PROFILE != "quality":
        logger.warning(f"未知 OCR_PROFILE={OCR_PROFILE}，改用 quality")
    OCR_PROFILE = "quality"
    DET_MODEL_URL, REC_MODEL_URL = _QUALITY_DET_URL, _QUALITY_REC_URL

SLM_GGUF_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

DET_PATH = os.path.join(MODEL_DIR, DET_MODEL_URL.rsplit("/", 1)[-1])
REC_PATH = os.path.join(MODEL_DIR, REC_MODEL_URL.rsplit("/", 1)[-1])
SLM_PATH = os.environ.get("SLM_MODEL_PATH", os.path.join(MODEL_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf"))

THREADS_INTRA = int(os.environ.get("OCR_INTRA_THREADS", "2"))
THREADS_INTER = int(os.environ.get("OCR_INTER_THREADS", "1"))
DET_BOX_THRESH = float(os.environ.get("DET_BOX_THRESH", "0.5"))
DET_UNCLIP_RATIO = float(os.environ.get("DET_UNCLIP_RATIO", "1.6"))
# 檢測輸入最長邊：960 係準確率/速度平衡點；736 快約 40% 但可能漏細字
DET_LIMIT_SIDE_LEN = int(os.environ.get("DET_LIMIT_SIDE_LEN", "960"))


def _download(url: str, path: str, retries: int = 3):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"📦 下載模型 ({attempt}/{retries}): {os.path.basename(path)}")
            urllib.request.urlretrieve(url, path)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return
        except Exception as e:
            wait = 2 ** attempt
            logger.warning(f"下載失敗（{e}），{wait}s 後重試…")
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            time.sleep(wait)
    raise RuntimeError(f"模型下載失敗，請檢查網絡後重試：{url}")


class HandwritingDualEngine:
    """PP-OCRv5 全頁引擎 + v5 rec-only 增強裁片擂台。"""

    def __init__(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        _download(DET_MODEL_URL, DET_PATH)
        _download(REC_MODEL_URL, REC_PATH)

        # SLM 預設唔下載；ENABLE_SLM=1 先下載（約 400MB）。
        # SKIP_SLM_DOWNLOAD=1 可阻止自動下載（例如已手動放好模型檔）。
        self.slm_available = os.path.exists(SLM_PATH) and os.path.getsize(SLM_PATH) > 0
        if (os.environ.get("ENABLE_SLM", "0") == "1"
                and os.environ.get("SKIP_SLM_DOWNLOAD", "0") != "1"
                and not self.slm_available):
            try:
                _download(SLM_GGUF_URL, SLM_PATH)
                self.slm_available = True
            except Exception as e:
                logger.warning(f"SLM 模型下載失敗，語境潤飾將停用：{e}")

        logger.info(f"🔧 載入 PP-OCRv5 全頁引擎（det+rec, profile={OCR_PROFILE}）…")
        self.full_engine = RapidOCR(
            det_model_path=DET_PATH,
            rec_model_path=REC_PATH,
            use_cls=False,                      # 唔做方向分類，慳 RAM
            intra_op_num_threads=THREADS_INTRA,
            inter_op_num_threads=THREADS_INTER,
            det_limit_side_len=DET_LIMIT_SIDE_LEN,
            det_box_thresh=DET_BOX_THRESH,      # 手寫偏淡可調低（如 0.4）
            det_unclip_ratio=DET_UNCLIP_RATIO,  # 手寫筆畫邊緣可調大（如 2.0）
        )

        logger.info("🔧 載入 PP-OCRv5 rec-only 引擎（增強裁片擂台用）…")
        # TextRecognizer 直接構造 rec-only：從源頭就冇 det，唔會嘥 RAM/CPU。
        # 字符表由 ONNX metadata 自動讀取（v5 模型內嵌）。
        self.rec_engine = TextRecognizer({
            "model_path": REC_PATH,
            "rec_img_shape": [3, 48, 320],
            "rec_batch_num": 6,
            "intra_op_num_threads": THREADS_INTRA,
            "inter_op_num_threads": THREADS_INTER,
        })
        logger.info("✅ 雙通道引擎載入完成")

    def _race_line(self, img_bgr: np.ndarray, box, base_text: str, base_score: float):
        """
        增強裁片擂台：透視校正 → 細裁片預放大 → CLAHE → v5 rec-only，
        分數贏過原結果先取代。任何一步出錯就保留原結果。
        """
        try:
            crop = AdaptiveImageTuner.crop_polygon_perspective(
                img_bgr, box, margin_ratio=0.12
            )
            if crop is None or crop.size == 0:
                return base_text, base_score, "v5-full"
            # 手寫行裁片通常好細；rec 內部會縮到高 48，預先放大保留筆畫細節
            crop = AdaptiveImageTuner.upscale_crop_for_rec(crop)
            crop = AdaptiveImageTuner.clahe_enhance(crop)
            rec_res, _ = self.rec_engine(crop)
            if not rec_res:
                return base_text, base_score, "v5-full"
            new_text, new_score = rec_res[0][0], float(rec_res[0][1])
            new_text = (new_text or "").strip()
            if new_text and new_score > base_score:
                return new_text, new_score, "v5-rec-enhanced"
        except Exception as e:
            logger.debug(f"擂台裁片失敗，保留原結果：{e}")
        return base_text, base_score, "v5-full"

    def process_image(self, img_bgr: np.ndarray, handwriting_threshold: float = 0.85):
        """
        輸入：BGR numpy 圖片
        輸出：([(box, text, score), ...], avg_score)
        handwriting_threshold：低過呢個分數嘅行先打擂台；
          設 1.0 即係每行都打（手寫文件建議，慢少少但準啲）。
        """
        if img_bgr is None or img_bgr.size == 0:
            return [], 0.0

        enhanced = AdaptiveImageTuner.auto_enhance_page(img_bgr)
        full_results, _ = self.full_engine(enhanced)
        if not full_results:
            return [], 0.0

        final_results = []
        for box, text, score in full_results:
            if score < handwriting_threshold:
                text, score, _ = self._race_line(enhanced, box, text, float(score))
            final_results.append((box, text, float(score)))

        avg_score = sum(r[2] for r in final_results) / len(final_results)
        return final_results, avg_score
