"""
模組 4：Qwen2.5-0.5B Subprocess 隔離推論 worker

由 main.py 以「獨立子進程」呼叫（python3 -m core.slm_worker），
推論完成進程即退出，OS 即時回收全部記憶體，唔會長期霸住 2GB RAM。

安全降級：
- llama-cpp-python 未安裝 → 原樣輸出（passthrough）
- GGUF 模型唔存在 → 原樣輸出
任何情況都唔會擲 exception 拖死主服務。
"""
import os
import sys

MODEL_PATH = os.environ.get(
    "SLM_MODEL_PATH", "./models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
MAX_INPUT_CHARS = int(os.environ.get("SLM_MAX_INPUT_CHARS", "3000"))
N_THREADS = int(os.environ.get("SLM_THREADS", "2"))


def main():
    input_text = sys.stdin.read().strip()
    if not input_text:
        return
    # 2GB 保險掣：輸入太長直接截斷，唔好畀 prompt 爆 context
    if len(input_text) > MAX_INPUT_CHARS:
        input_text = input_text[:MAX_INPUT_CHARS]

    try:
        from llama_cpp import Llama
    except ImportError:
        sys.stdout.write(input_text)
        return

    if not os.path.exists(MODEL_PATH):
        sys.stdout.write(input_text)
        return

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=N_THREADS,
        n_batch=256,
        verbose=False,
    )

    prompt = (
        "<|im_start|>system\n"
        "你是一個香港商業文件的專業 OCR 後處理校正器。請修正文字中的錯別字、形近字與斷字。\n"
        "手寫體常見混淆僅作參考（未/末、己/已/巳、人/入、大/太、千/干、土/士、"
        "0/O、1/l/I、5/S、8/B），必須結合上下文先好改，唔好硬改。\n"
        "規則：嚴禁回答問題或添加解釋，嚴禁改動原文格式與換行，"
        "必須保留廣東話專用字元，直接輸出校正後的純文本。<|im_end|>\n"
        f"<|im_start|>user\n{input_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    try:
        output = llm(
            prompt,
            max_tokens=min(len(input_text) * 2 + 32, 1024),
            temperature=0.0,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        res = output["choices"][0]["text"].strip()
    except Exception:
        res = ""

    sys.stdout.write(res if res else input_text)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
