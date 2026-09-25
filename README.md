# DocuOCR Enterprise Pro — ARM64 + 2GB RAM 特化版

香港商業文件 OCR：PP-OCRv5 雙通道（全頁 + 增強裁片擂台）、手寫中英文強化、
OpenCC s2hk + 香港字典 + SymSpell + 在線反饋學習，輸出可搜尋雙層 PDF。

## 檔案結構

```text
├── core/
│   ├── adaptive_image.py   # 模組2：自適應銳化、透視校正、局部 CLAHE
│   ├── fuzzy_corrector.py  # 模組3：SymSpell 英文模糊校正
│   ├── hk_postprocess.py   # 模組1：OpenCC s2hk + 反饋字典熱重載
│   ├── slm_worker.py       # 模組4：Qwen2.5-0.5B 子進程隔離推論（可選）
│   ├── ocr_engine.py       # 雙 ONNX 引擎（2GB 線程鎖定）
│   └── pdf_renderer.py     # 雙層 PDF + malloc_trim
├── lexicons/               # 香港字典（可直接喺 NAS 改，即時生效）
├── static/index.html       # Web UI（PDF + 單圖 + 反饋學習）
├── models/                 # 模型（自動下載）
├── main.py                 # FastAPI 主服務
├── requirements.txt        # 核心依賴（全部有 ARM64 wheel，免編譯）
├── requirements-slm.txt    # 可選：llama-cpp-python（ARM 需編譯 20–40 分鐘）
├── Dockerfile
└── docker-compose.yml
```

## 部署（Synology Container Manager）

1. 喺 File Station 建立 `/docker/docuocr-enterprise/`，將本專案所有檔案上傳入去。
2. Container Manager → 專案 → 新增 → 專案名稱 `docuocr` → 路徑揀 `/docker/docuocr-enterprise/` →
   來源「從路徑建立」→ 佢會自動讀取 `docker-compose.yml` → 下一步 → 完成。
3. 首次啟動會自動下載模型（約 200MB：PP-OCRv5 det server + rec server，RapidAI 官方 ModelScope 鏡像），之後重用唔使再下。
4. 開啟 `http://<NAS-IP>:8000` 即用。`/health` 可檢查狀態。

或用 SSH：

```bash
cd /volume1/docker/docuocr-enterprise
docker compose up -d --build
docker logs -f docuocr-enterprise
```

## 方法二：直接 pull 預建 ARM64 鏡像（推薦，唔使喺 NAS 上 build）

1. 將本專案 push 去你嘅 GitHub repo（`main` 分支），`.github/workflows/build-arm64.yml`
   會自動用 QEMU 建置 `linux/arm64` 鏡像並推送去 GHCR：
   - `ghcr.io/<用戶名>/docuocr-enterprise:latest` —— 基礎版（NAS 用呢個）
   - `ghcr.io/<用戶名>/docuocr-enterprise:slm` —— 連 llama-cpp（Qwen2.5-0.5B 已編譯好）
2. 去 GitHub → 右上頭像 → Packages → 將 `docuocr-enterprise` 設為 **Public**，
   否則 NAS 未登入會 pull 唔到（替代方案：喺 NAS 上 `docker login ghcr.io` 用 PAT 登入）。
3. 照 `docker-compose.yml` 內「方法二」註釋改好，執行：
   ```bash
   docker compose pull && docker compose up -d
   ```
4. 之後每次更新程式碼 push 上 GitHub，Actions 會自動出新鏡像，
   NAS 端 `docker compose pull && docker compose up -d` 就升級完。

## 記憶體實測預期（2GB 上限內）

| 狀態 | 約數 |
|---|---|
| Idle（引擎懶加載，未有請求） | ~150–250 MB |
| OCR 推論中（單併發） | ~900–1300 MB |
| Qwen2.5-0.5B 子進程推論峰值 | ~+400 MB（完成即 100% 釋放） |

## API

| 方法 | 路徑 | 說明 |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/health` | 狀態（含架構、引擎/SLM 就緒） |
| POST | `/api/v1/ocr/pdf` | PDF → 雙層可搜尋 PDF（multipart form） |
| POST | `/api/v1/ocr/image` | 單張圖片 → JSON（text / score / box） |
| POST | `/api/v1/feedback` | `{"wrong_text","correct_text"}` 在線學習 |

## 環境變數（docker-compose.yml 可改）

| 變數 | 預設 | 說明 |
|---|---|---|
| `MAX_CONCURRENT_JOBS` | `1` | 併發任務數；2GB 機唔好調大 |
| `ENABLE_SLM` | `0` | 開關 Qwen2.5-0.5B 語境潤飾 |
| `SKIP_SLM_DOWNLOAD` | `1` | 唔下載 400MB GGUF |
| `OCR_INTRA_THREADS` / `OCR_INTER_THREADS` | `2` / `1` | onnxruntime 線程數 |
| `MAX_PAGE_DIM` | `3500` | 頁面像素邊長上限，超咗自動降 DPI |
| `NATIVE_TEXT_MIN_CHARS` | `50` | 原生文字頁判定門檻 |

## 可選：啟用 Qwen2.5-0.5B（SLM）

1. 解開 `Dockerfile` 內 SLM 兩行，`docker compose up -d --build`（ARM 編譯約 20–40 分鐘）。
2. `docker-compose.yml`：`ENABLE_SLM=1`、`SKIP_SLM_DOWNLOAD=0`，重啟後自動下載 GGUF。
3. Web UI 剔選「🧠 語境潤飾」或 API 傳 `enable_slm=true`。

## 舊版 bug 修正一覽（已驗證源碼）

1. `rm_session_options` 係無效參數（傳入去冇任何作用）→ 改用正確嘅
   `intra_op_num_threads` / `inter_op_num_threads`，並會 propagate 去 Det/Rec。
2. 手寫引擎 `det_use=False` 寫錯（正確係 `use_det=False`），舊寫法根本冇關到 det →
   而家 rec-only 直接用 `TextRecognizer` 構造，從源頭就冇 det，慳返檢測模型嘅 RAM/CPU。
3. 新增 `use_cls=False`：唔載入文字方向分類模型。

## 手寫中英文調優指南

預設已經係手寫強化版（PP-OCRv5：官方話大幅改善連筆/非規範手寫；單模型覆蓋繁中+英文）。
如果份文件以手寫為主，仲可以咁樣榨多啲準確率：

| 手段 | 做法 | 效果 |
|---|---|---|
| 每行都打擂台 | request 傳 `handwriting_threshold=1.0`（預設 0.85） | 低分行先會用增強裁片再認；設 1.0 即每行都做，慢少少但準啲 |
| 捉淡色筆跡 | `docker-compose.yml` 解開 `DET_BOX_THRESH=0.4` | 檢測閾值降低，鉛筆/淺色原子筆都捉到 |
| 包埋筆畫邊緣 | `docker-compose.yml` 解開 `DET_UNCLIP_RATIO=2.0` | 檢測框放大，唔會切走筆畫收尾 |
| 用 300 DPI | PDF 掃描/渲染用 300dpi | 手寫細字質素明顯提升 |

為何唔用 TrOCR 等 Transformer 手寫模型：佢哋喺 2GB ARM 上跑一頁要幾分鐘，
RAM 亦爆錶；PP-OCRv5 server 係同級硬件下準確率/速度最平衡嘅選擇。

## 效能調優（2 核 + 2GB ARM）

已經內建嘅慳資源措施：單 worker、一次只做一個 job（`MAX_CONCURRENT_JOBS=1`）、
`use_cls=False`（唔載方向分類模型）、onnxruntime `intra_op=2`/`inter_op=1`、
`enable_cpu_mem_arena=False`（rapidocr 預設，唔畀 arena 無限脹）、
`OMP_NUM_THREADS=2` + `OPENBLAS_NUM_THREADS=1`（唔畀 numpy 同 onnxruntime 爭 CPU）、
每頁處理完 `gc.collect()` + `malloc_trim()` 還記憶體畀 OS、超大頁自動降 DPI（`MAX_PAGE_DIM=3500`）。

### 兩個速度檔：quality vs fast

| | quality（預設） | fast |
|---|---|---|
| 模型 | PP-OCRv5 server（det+rec） | PP-OCRv5 mobile（det ~5MB / rec ~17MB） |
| 開法 | 唔使搞（預設） | `OCR_PROFILE=fast` |
| 速度 | 基準 | 檢測快幾倍，成頁快約 2–4 倍（估算） |
| 準確率 | 最高，尤其手寫/細字 | 印刷體大字 OK，手寫/細字明顯差啲 |
| RAM | 較高 | 更低 |

`fast` 適合：純印刷體、字唔細、要追速度。手寫文件請留喺 `quality`。

### 其他掣

- `DET_LIMIT_SIDE_LEN=736`（預設 960）：檢測輸入縮細，加速約 40%，代價係可能漏細字。
- `PREWARM_ON_STARTUP=1`（預設 0）：開機後背景預載模型，第一次 request 唔使等多 30 秒；
  代價係開機頭一分鐘 CPU 較忙。
- PDF 嘅 `dpi` 參數：200 係平衡點；150 更快但細字差，300 慢但手寫好。
- `SKIP_SLM_DOWNLOAD` / `ENABLE_SLM`：SLM 語境潤飾每頁慢幾秒，2GB 機預設關閉。

> 以上倍數係估算：ARM NAS 效能差異極大（由 Realtek 到 RK3588），
> 實際請喺你部機度試一頁計時再決定。
4. `OpenCC`（要編譯 C++）→ `opencc-python-reimplemented`（純 Python，ARM 免編譯）。
5. 刪除根本冇用到嘅 `scikit-image`（慳幾百 MB 鏡像體積）。
6. `pkg_resources`（已棄用）→ `importlib.resources`。
7. `llama-cpp-python` 移出預設依賴（PyPI 冇 aarch64 wheel，唔好拖慢 build）。
8. 反饋寫檔改原子寫入（`os.replace`），加 `threading.Lock` 防 race。
