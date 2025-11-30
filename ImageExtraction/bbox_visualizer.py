"""バウンディングボックスを画像に描画して可視化する。"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

from .pdf_renderer import BBox
from .layout_analyzer import LayoutElement, ElementType
from .figure_clusterer import FigureObject


class BBoxVisualizer:
    """バウンディングボックスを画像上に描画するクラス。"""
    
    # 要素タイプごとの色設定（RGB）
    TYPE_COLORS = {
        ElementType.TEXT: (255, 0, 0),      # 赤
        ElementType.FIGURE: (0, 255, 0),    # 緑
        ElementType.TABLE: (0, 0, 255),     # 青
        ElementType.TITLE: (255, 255, 0),   # 黄
        ElementType.LIST: (255, 0, 255),    # マゼンタ
    }
    
    def __init__(
        self,
        line_width: int = 3,
        font_size: int = 20,
        show_labels: bool = True
    ) -> None:
        """初期化。
        
        Args:
            line_width: ボックスの線幅
            font_size: ラベルのフォントサイズ
            show_labels: ラベル（要素タイプ、信頼度）を表示するか
        """
        self.line_width = line_width
        self.font_size = font_size
        self.show_labels = show_labels
        self._font = None
    
    def _get_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """フォントを取得（キャッシュ）。"""
        if self._font is None:
            try:
                # システムフォントを試す
                self._font = ImageFont.truetype("arial.ttf", self.font_size)
            except OSError:
                # フォールバック: デフォルトフォント
                self._font = ImageFont.load_default()
        return self._font
    
    def draw_layout_elements(
        self,
        image: bytes | Image.Image,
        elements: List[LayoutElement],
        output_path: Path | str
    ) -> None:
        """レイアウト要素のバウンディングボックスを描画して保存。
        
        Args:
            image: 元画像（PNG/JPEGバイト列またはPIL Image）
            elements: 描画するレイアウト要素のリスト
            output_path: 出力先ファイルパス
        """
        # PIL Imageに変換
        if isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image.copy()
        
        # RGB変換（描画用）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        draw = ImageDraw.Draw(pil_image)
        
        for i, elem in enumerate(elements):
            color = self.TYPE_COLORS.get(elem.element_type, (128, 128, 128))
            
            # バウンディングボックスを描画
            self._draw_box(draw, elem.bbox, color)
            
            # ラベルを描画
            if self.show_labels:
                label = f"{elem.element_type.value} ({elem.confidence:.2f})"
                self._draw_label(draw, elem.bbox, label, color)
        
        # 保存
        pil_image.save(output_path)
    
    def draw_figure_objects(
        self,
        image: bytes | Image.Image,
        figures: List[FigureObject],
        output_path: Path | str
    ) -> None:
        """図オブジェクトのバウンディングボックスを描画して保存。
        
        Args:
            image: 元画像（PNG/JPEGバイト列またはPIL Image）
            figures: 描画する図オブジェクトのリスト
            output_path: 出力先ファイルパス
        """
        # PIL Imageに変換
        if isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image.copy()
        
        # RGB変換（描画用）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        draw = ImageDraw.Draw(pil_image)
        
        for i, fig in enumerate(figures):
            # 図オブジェクト全体を青で描画
            color = (0, 128, 255)
            self._draw_box(draw, fig.bbox, color, width=self.line_width + 2)
            
            # ラベルを描画
            if self.show_labels:
                label = f"Fig{i+1} ({fig.confidence:.2f}, {len(fig.elements)}elem)"
                self._draw_label(draw, fig.bbox, label, color)
            
            # 内部の要素を薄い色で描画
            for elem in fig.elements:
                elem_color = self.TYPE_COLORS.get(elem.element_type, (200, 200, 200))
                # 半透明風に薄くする
                elem_color_light = tuple(int(c * 0.5 + 255 * 0.5) for c in elem_color)
                self._draw_box(draw, elem.bbox, elem_color_light, width=1)
        
        # 保存
        pil_image.save(output_path)
    
    def draw_bboxes(
        self,
        image: bytes | Image.Image,
        bboxes: List[Tuple[BBox, str]],
        output_path: Path | str,
        color: Tuple[int, int, int] = (255, 0, 0)
    ) -> None:
        """汎用的なバウンディングボックスリストを描画。
        
        Args:
            image: 元画像（PNG/JPEGバイト列またはPIL Image）
            bboxes: (bbox, label)のリスト
            output_path: 出力先ファイルパス
            color: 描画色（RGB）
        """
        # PIL Imageに変換
        if isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image.copy()
        
        # RGB変換（描画用）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        draw = ImageDraw.Draw(pil_image)
        
        for bbox, label in bboxes:
            self._draw_box(draw, bbox, color)
            if self.show_labels and label:
                self._draw_label(draw, bbox, label, color)
        
        # 保存
        pil_image.save(output_path)
    
    def _draw_box(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: BBox,
        color: Tuple[int, int, int],
        width: int | None = None
    ) -> None:
        """バウンディングボックスを描画。"""
        if width is None:
            width = self.line_width
        
        draw.rectangle(
            [(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)],
            outline=color,
            width=width
        )
    
    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: BBox,
        text: str,
        color: Tuple[int, int, int]
    ) -> None:
        """ラベルテキストを描画。"""
        font = self._get_font()
        
        # テキストの背景を描画（視認性向上）
        # bbox の左上に配置
        text_x = bbox.x0
        text_y = bbox.y0 - self.font_size - 4
        
        # 画像外に出ないように調整
        if text_y < 0:
            text_y = bbox.y0 + 2
        
        # 背景矩形
        try:
            text_bbox = draw.textbbox((text_x, text_y), text, font=font)
            draw.rectangle(text_bbox, fill=(255, 255, 255, 200))
        except AttributeError:
            # 古いPillowバージョンではtextbboxがない
            pass
        
        # テキスト描画
        draw.text((text_x, text_y), text, fill=color, font=font)
