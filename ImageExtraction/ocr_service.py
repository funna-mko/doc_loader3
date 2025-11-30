"""画像内のテキストをOCRで抽出するサービス。"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional

from PIL import Image

from .pdf_renderer import BBox


@dataclass
class OcrResult:
    """OCR結果。"""
    
    text: str
    bbox: BBox
    confidence: float  # 0.0 - 1.0


class OcrService:
    """画像からテキストをOCR抽出するサービス。
    
    pytesseract または EasyOCR を使用可能。
    """
    
    def __init__(
        self,
        engine: str = "tesseract",
        lang: str = "jpn+eng",
        tesseract_cmd: Optional[str] = None
    ) -> None:
        """初期化。
        
        Args:
            engine: OCRエンジン ("tesseract" または "easyocr")
            lang: 認識言語 (tesseract: "jpn+eng", easyocr: ["ja", "en"])
            tesseract_cmd: tesseractコマンドのパス（Noneの場合は自動検出）
        """
        self.engine = engine
        self.lang = lang
        self.tesseract_cmd = tesseract_cmd
        self._ocr_instance: Optional[object] = None
    
    def run(self, image: bytes | Image.Image) -> List[OcrResult]:
        """画像からテキストをOCR抽出する。
        
        Args:
            image: PNG/JPEGバイト列またはPIL Image
            
        Returns:
            OCR結果のリスト（テキスト、バウンディングボックス、信頼度）
        """
        # PIL Imageに変換
        if isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image
        
        if self.engine == "tesseract":
            return self._run_tesseract(pil_image)
        elif self.engine == "easyocr":
            return self._run_easyocr(pil_image)
        else:
            raise ValueError(f"未サポートのOCRエンジン: {self.engine}")
    
    def _run_tesseract(self, image: Image.Image) -> List[OcrResult]:
        """pytesseractでOCR実行。"""
        try:
            import pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract がインストールされていません。"
                "pip install pytesseract でインストールしてください。"
            )
        
        # tesseract_cmdが指定されている場合は設定
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        
        # データフレーム形式で詳細情報を取得
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            output_type=pytesseract.Output.DICT
        )
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
            
            # 信頼度が低いものはスキップ
            conf = float(data['conf'][i])
            if conf < 0:
                continue
            
            x = float(data['left'][i])
            y = float(data['top'][i])
            w = float(data['width'][i])
            h = float(data['height'][i])
            
            bbox = BBox(
                x0=x,
                y0=y,
                x1=x + w,
                y1=y + h
            )
            
            results.append(OcrResult(
                text=text,
                bbox=bbox,
                confidence=conf / 100.0  # 0-100 を 0.0-1.0 に変換
            ))
        
        return results
    
    def _run_easyocr(self, image: Image.Image) -> List[OcrResult]:
        """EasyOCRでOCR実行。"""
        try:
            import easyocr
            import numpy as np
        except ImportError:
            raise ImportError(
                "easyocr がインストールされていません。"
                "pip install easyocr でインストールしてください。"
            )
        
        # EasyOCRリーダーを初期化（キャッシュ）
        if self._ocr_instance is None:
            # lang を文字列からリストに変換
            if isinstance(self.lang, str):
                lang_list = [l.strip() for l in self.lang.replace('+', ',').split(',')]
                # tesseract形式からeasyocr形式に変換
                lang_map = {'jpn': 'ja', 'eng': 'en'}
                lang_list = [lang_map.get(l, l) for l in lang_list]
            else:
                lang_list = self.lang
            
            self._ocr_instance = easyocr.Reader(lang_list)
        
        # numpy配列に変換
        img_array = np.array(image)
        
        # OCR実行
        results_raw = self._ocr_instance.readtext(img_array)
        
        results = []
        for detection in results_raw:
            bbox_coords, text, conf = detection
            
            # バウンディングボックスの座標を抽出（4点 -> 矩形）
            xs = [pt[0] for pt in bbox_coords]
            ys = [pt[1] for pt in bbox_coords]
            
            bbox = BBox(
                x0=float(min(xs)),
                y0=float(min(ys)),
                x1=float(max(xs)),
                y1=float(max(ys))
            )
            
            results.append(OcrResult(
                text=text,
                bbox=bbox,
                confidence=float(conf)
            ))
        
        return results
    
    def extract_from_region(
        self,
        image: bytes | Image.Image,
        region: BBox
    ) -> str:
        """画像の特定領域からテキストを抽出。
        
        Args:
            image: 元画像
            region: 抽出する領域のバウンディングボックス
            
        Returns:
            抽出されたテキスト（改行で結合）
        """
        # PIL Imageに変換
        if isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image
        
        # 領域を切り出し
        cropped = pil_image.crop((region.x0, region.y0, region.x1, region.y1))
        
        # OCR実行
        results = self.run(cropped)
        
        # テキストを結合
        return '\n'.join(r.text for r in results)
