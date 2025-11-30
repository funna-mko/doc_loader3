"""ページ画像からレイアウト要素（図・表・テキストブロック）を検出する。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
from PIL import Image

from .pdf_renderer import BBox


class ElementType(Enum):
    """レイアウト要素の種類。"""
    
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"
    TITLE = "title"
    LIST = "list"


@dataclass
class LayoutElement:
    """検出されたレイアウト要素。"""
    
    bbox: BBox
    element_type: ElementType
    confidence: float  # 0.0 - 1.0


class LayoutAnalyzer:
    """ページ画像からレイアウト要素を検出する。
    
    現在はシンプルなルールベースの実装。将来的にはDetectron2等のモデルに置き換え可能。
    """
    
    def __init__(self, min_area: int = 1000) -> None:
        """初期化。
        
        Args:
            min_area: 要素として認識する最小面積（ピクセル^2）
        """
        self.min_area = min_area
    
    def detect(self, image: bytes | Image.Image) -> List[LayoutElement]:
        """画像からレイアウト要素を検出する。
        
        Args:
            image: PNG/JPEG等のバイト列、またはPIL Image
            
        Returns:
            検出された要素のリスト
        """
        if isinstance(image, bytes):
            from io import BytesIO
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image
        
        # グレースケールに変換
        gray = pil_image.convert('L')
        img_array = np.array(gray)
        
        # 簡易的な二値化（閾値ベース）
        # 白い背景（高輝度）と黒い内容（低輝度）を分離
        threshold = 240
        binary = img_array < threshold
        
        # 連結成分解析で候補領域を抽出
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)
        
        elements = []
        for label_id in range(1, num_features + 1):
            # ラベルごとの座標を取得
            coords = np.argwhere(labeled == label_id)
            if len(coords) == 0:
                continue
            
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 面積チェック
            area = (x_max - x_min) * (y_max - y_min)
            if area < self.min_area:
                continue
            
            bbox = BBox(
                x0=float(x_min),
                y0=float(y_min),
                x1=float(x_max),
                y1=float(y_max)
            )
            
            # 簡易的な要素タイプ判定
            # アスペクト比で判断（仮実装）
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / height if height > 0 else 0
            
            if aspect_ratio > 3.0:
                # 横長 → テキスト行の可能性
                elem_type = ElementType.TEXT
                confidence = 0.7
            elif 0.5 < aspect_ratio < 2.0:
                # 正方形に近い → 図の可能性
                elem_type = ElementType.FIGURE
                confidence = 0.6
            else:
                # 縦長 → リストや表の可能性
                elem_type = ElementType.TABLE
                confidence = 0.5
            
            elements.append(LayoutElement(
                bbox=bbox,
                element_type=elem_type,
                confidence=confidence
            ))
        
        return elements
    
    def align_text(
        self,
        elements: List[LayoutElement],
        text_blocks: List[tuple[str, BBox]]
    ) -> List[tuple[LayoutElement, str]]:
        """レイアウト要素とテキストブロックを座標で紐付ける。
        
        Args:
            elements: 検出されたレイアウト要素
            text_blocks: (テキスト, bbox) のリスト
            
        Returns:
            (要素, 対応テキスト) のリスト
        """
        results = []
        
        for elem in elements:
            matched_text = ""
            for text, text_bbox in text_blocks:
                # バウンディングボックスの重なりをチェック
                if self._boxes_overlap(elem.bbox, text_bbox):
                    matched_text += text + " "
            
            results.append((elem, matched_text.strip()))
        
        return results
    
    @staticmethod
    def _boxes_overlap(bbox1: BBox, bbox2: BBox) -> bool:
        """2つのバウンディングボックスが重なるかチェック。"""
        return not (
            bbox1.x1 < bbox2.x0 or
            bbox1.x0 > bbox2.x1 or
            bbox1.y1 < bbox2.y0 or
            bbox1.y0 > bbox2.y1
        )
