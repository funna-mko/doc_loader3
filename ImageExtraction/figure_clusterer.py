"""バウンディングボックスをグルーピングして図オブジェクトとしてまとめる。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set

from .layout_analyzer import LayoutElement, ElementType
from .pdf_renderer import BBox


@dataclass
class FigureObject:
    """グルーピングされた図オブジェクト。"""
    
    bbox: BBox
    elements: List[LayoutElement] = field(default_factory=list)
    confidence: float = 0.0
    
    def merge(self, other: FigureObject) -> FigureObject:
        """他の図オブジェクトとマージする。
        
        Args:
            other: マージする図オブジェクト
            
        Returns:
            マージ後の新しい図オブジェクト
        """
        # バウンディングボックスを統合
        new_bbox = BBox(
            x0=min(self.bbox.x0, other.bbox.x0),
            y0=min(self.bbox.y0, other.bbox.y0),
            x1=max(self.bbox.x1, other.bbox.x1),
            y1=max(self.bbox.y1, other.bbox.y1)
        )
        
        # 要素を結合
        new_elements = self.elements + other.elements
        
        # 信頼度は平均を取る
        new_confidence = (self.confidence + other.confidence) / 2
        
        return FigureObject(
            bbox=new_bbox,
            elements=new_elements,
            confidence=new_confidence
        )


class FigureClusterer:
    """レイアウト要素をグルーピングして図オブジェクトにまとめる。"""
    
    def __init__(
        self,
        distance_threshold: float = 50.0,
        overlap_threshold: float = 0.3
    ) -> None:
        """初期化。
        
        Args:
            distance_threshold: 要素間の最大距離（ピクセル）。この距離以内なら同一図と判定
            overlap_threshold: 重なり率の閾値。この比率以上重なっていたら同一図と判定
        """
        self.distance_threshold = distance_threshold
        self.overlap_threshold = overlap_threshold
    
    def group(self, elements: List[LayoutElement]) -> List[FigureObject]:
        """レイアウト要素をグルーピングする。
        
        Args:
            elements: レイアウト要素のリスト
            
        Returns:
            グルーピングされた図オブジェクトのリスト
        """
        if not elements:
            return []
        
        # 初期状態: 各要素を個別の図オブジェクトとして扱う
        figures = [
            FigureObject(
                bbox=elem.bbox,
                elements=[elem],
                confidence=elem.confidence
            )
            for elem in elements
        ]
        
        # 反復的にマージ
        merged = True
        while merged:
            merged = False
            new_figures = []
            used_indices: Set[int] = set()
            
            for i in range(len(figures)):
                if i in used_indices:
                    continue
                
                current = figures[i]
                
                # 他の図との距離・重なりをチェック
                for j in range(i + 1, len(figures)):
                    if j in used_indices:
                        continue
                    
                    other = figures[j]
                    
                    if self._should_merge(current.bbox, other.bbox):
                        # マージ
                        current = current.merge(other)
                        used_indices.add(j)
                        merged = True
                
                new_figures.append(current)
                used_indices.add(i)
            
            figures = new_figures
        
        return figures
    
    def _should_merge(self, bbox1: BBox, bbox2: BBox) -> bool:
        """2つのバウンディングボックスをマージすべきか判定。
        
        Args:
            bbox1: 最初のバウンディングボックス
            bbox2: 2番目のバウンディングボックス
            
        Returns:
            マージすべきならTrue
        """
        # 重なりチェック
        overlap = self._calculate_overlap(bbox1, bbox2)
        if overlap >= self.overlap_threshold:
            return True
        
        # 距離チェック
        distance = self._calculate_distance(bbox1, bbox2)
        if distance <= self.distance_threshold:
            return True
        
        return False
    
    @staticmethod
    def _calculate_overlap(bbox1: BBox, bbox2: BBox) -> float:
        """2つのバウンディングボックスの重なり率を計算。
        
        Returns:
            0.0（重なりなし）から1.0（完全一致）の値
        """
        # 重なり領域を計算
        x_overlap = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0))
        y_overlap = max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))
        overlap_area = x_overlap * y_overlap
        
        if overlap_area == 0:
            return 0.0
        
        # 各ボックスの面積
        area1 = (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0)
        area2 = (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0)
        
        # 小さい方の面積に対する重なり率
        min_area = min(area1, area2)
        return overlap_area / min_area if min_area > 0 else 0.0
    
    @staticmethod
    def _calculate_distance(bbox1: BBox, bbox2: BBox) -> float:
        """2つのバウンディングボックス間の距離を計算。
        
        Returns:
            中心点間のユークリッド距離
        """
        # 中心点を計算
        center1_x = (bbox1.x0 + bbox1.x1) / 2
        center1_y = (bbox1.y0 + bbox1.y1) / 2
        center2_x = (bbox2.x0 + bbox2.x1) / 2
        center2_y = (bbox2.y0 + bbox2.y1) / 2
        
        # ユークリッド距離
        dx = center2_x - center1_x
        dy = center2_y - center1_y
        return (dx * dx + dy * dy) ** 0.5
