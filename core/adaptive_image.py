"""
模組 2：自適應圖像增強 + 四點透視校正
- 全圖只做「模糊先銳化」，嚴禁全圖 CLAHE（會引致 PP-OCR Det 假框）
- CLAHE 只落喺裁出嚟嘅手寫小圖上
"""
import cv2
import numpy as np


class AdaptiveImageTuner:
    @staticmethod
    def auto_enhance_page(img_bgr: np.ndarray) -> np.ndarray:
        """根據模糊度動態銳化全圖"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 淨係當真係矇先做微量反差銳化
        if blur_score < 100.0:
            gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
            img_bgr = cv2.addWeighted(img_bgr, 1.4, gaussian, -0.4, 0)

        return img_bgr

    @staticmethod
    def upscale_crop_for_rec(crop: np.ndarray, min_h: int = 64,
                             target_h: int = 96, max_scale: float = 3.0) -> np.ndarray:
        """
        細裁片預放大（手寫行專用）。
        手寫行裁片通常得幾十 px 高；rec 模型內部會再縮到高 48，
        預先用 INTER_CUBIC 放大可以保留筆畫細節，明顯提升細字/手寫準確率。
        夠大嘅裁片原樣返回，唔嘥時間。
        """
        h = crop.shape[0]
        if h <= 0 or h >= min_h:
            return crop
        scale = min(target_h / h, max_scale)
        return cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def crop_polygon_perspective(
        img: np.ndarray, box: list, margin_ratio: float = 0.12
    ) -> np.ndarray | None:
        """四點透視校正 + 比例 Padding + 局部手寫 CLAHE"""
        try:
            pts = np.array(box, dtype=np.float32)

            # 1. 按比例向外擴展（防手寫連筆被切斷）
            center = np.mean(pts, axis=0)
            expanded = pts + (pts - center) * margin_ratio
            h_img, w_img = img.shape[:2]
            expanded[:, 0] = np.clip(expanded[:, 0], 0, w_img - 1)
            expanded[:, 1] = np.clip(expanded[:, 1], 0, h_img - 1)

            # 2. 排序四角：左上、右上、右下、左下
            rect = np.zeros((4, 2), dtype="float32")
            s = expanded.sum(axis=1)
            rect[0] = expanded[np.argmin(s)]
            rect[2] = expanded[np.argmax(s)]
            diff = np.diff(expanded, axis=1)
            rect[1] = expanded[np.argmin(diff)]
            rect[3] = expanded[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            max_w = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
            max_h = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
            if max_w <= 3 or max_h <= 3:
                return None

            dst = np.array(
                [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                dtype="float32",
            )

            # 3. 透視變換
            m = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(img, m, (max_w, max_h))

            # 4. CLAHE 只做喺手寫小圖（唔做全圖）
            gray_crop = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray_crop)
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        except Exception:
            return None
