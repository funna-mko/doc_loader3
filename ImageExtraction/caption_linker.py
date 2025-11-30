"""図オブジェクトと周辺テキストを紐付ける。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .figure_clusterer import FigureObject
from .pdf_renderer import BBox, TextBlock


@dataclass
class CaptionedFigure:
    """キャプション付き図オブジェクト。"""
    
    figure: FigureObject
    caption: Optional[str] = None
    nearby_text: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        if self.nearby_text is None:
            self.nearby_text = []


class CaptionLinker:
    """図オブジェクトと周辺テキストブロックを紐付ける。"""
    
    def __init__(
        self,
        caption_distance_threshold: float = 50.0,
        context_distance_threshold: float = 150.0
    ) -> None:
        """初期化。
        
        Args:
            caption_distance_threshold: キャプション判定の距離閾値（ピクセル）
            context_distance_threshold: 周辺テキスト判定の距離閾値（ピクセル）
        """
        self.caption_distance_threshold = caption_distance_threshold
        self.context_distance_threshold = context_distance_threshold
    
    def attach(
        self,
        figures: List[FigureObject],
        text_blocks: List[TextBlock]
    ) -> List[CaptionedFigure]:
        """図オブジェクトにテキストブロックを紐付ける。
        
        Args:
            figures: 図オブジェクトのリスト
            text_blocks: テキストブロックのリスト
            
        Returns:
            キャプション付き図オブジェクトのリスト
        """
        captioned_figures = []
        
        for fig in figures:
            caption = None
            caption_distance = float('inf')
            nearby_texts = []
            
            for block in text_blocks:
                distance = self._calculate_distance(fig.bbox, block.bbox)
                
                # キャプション判定（図の直下または直上にあるテキスト）
                if self._is_below_or_above(fig.bbox, block.bbox):
                    # 直下/直上にあり、最も近いものをキャプションとする
                    if distance < caption_distance:
                        caption = block.text
                        caption_distance = distance
                
                # 周辺テキスト判定
                if distance <= self.context_distance_threshold:
                    nearby_texts.append(block.text)
            
            captioned_figures.append(CaptionedFigure(
                figure=fig,
                caption=caption,
                nearby_text=nearby_texts
            ))
        
        return captioned_figures
    
    def fetch_context(
        self,
        figure: FigureObject,
        text_blocks: List[TextBlock],
        max_distance: float = 200.0
    ) -> List[TextBlock]:
        """図の周辺にあるテキストブロックを取得。
        
        Args:
            figure: 図オブジェクト
            text_blocks: テキストブロックのリスト
            max_distance: 最大距離（ピクセル）
            
        Returns:
            周辺のテキストブロックのリスト（距離順）
        """
        blocks_with_distance = []
        
        for block in text_blocks:
            distance = self._calculate_distance(figure.bbox, block.bbox)
            if distance <= max_distance:
                blocks_with_distance.append((distance, block))
        
        # 距離でソート
        blocks_with_distance.sort(key=lambda x: x[0])
        
        return [block for _, block in blocks_with_distance]
    
    @staticmethod
    def _calculate_distance(bbox1: BBox, bbox2: BBox) -> float:
        """2つのバウンディングボックス間の距離を計算。
        
        Returns:
            中心点間のユークリッド距離
        """
        center1_x = (bbox1.x0 + bbox1.x1) / 2
        center1_y = (bbox1.y0 + bbox1.y1) / 2
        center2_x = (bbox2.x0 + bbox2.x1) / 2
        center2_y = (bbox2.y0 + bbox2.y1) / 2
        
        dx = center2_x - center1_x
        dy = center2_y - center1_y
        return (dx * dx + dy * dy) ** 0.5
    
    @staticmethod
    def _is_below_or_above(fig_bbox: BBox, text_bbox: BBox) -> bool:
        """テキストが図の直下または直上にあるか判定。
        
        Args:
            fig_bbox: 図のバウンディングボックス
            text_bbox: テキストのバウンディングボックス
            
        Returns:
            直下または直上ならTrue
        """
        # 水平方向の重なりをチェック
        horizontal_overlap = not (
            text_bbox.x1 < fig_bbox.x0 or
            text_bbox.x0 > fig_bbox.x1
        )
        
        if not horizontal_overlap:
            return False
        
        # 垂直方向の位置関係をチェック
        is_below = text_bbox.y0 >= fig_bbox.y1
        is_above = text_bbox.y1 <= fig_bbox.y0
        
        return is_below or is_above
    
    @staticmethod
    def _find_block_by_text(blocks: List[TextBlock], text: str) -> Optional[TextBlock]:
        """テキスト内容からブロックを検索。"""
        for block in blocks:
            if block.text == text:
                return block
        return None
