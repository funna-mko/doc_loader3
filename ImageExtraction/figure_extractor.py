"""図オブジェクトを画像として切り出して保存する。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from .caption_linker import CaptionedFigure
from .pdf_renderer import BBox


@dataclass
class FigureMetadata:
    """切り出された図のメタデータ。"""
    
    image_path: str
    page_number: int
    bbox: Dict[str, float]  # x0, y0, x1, y1
    caption: Optional[str] = None
    linked_text: Optional[List[str]] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """辞書形式に変換。"""
        return asdict(self)


class FigureExtractor:
    """図オブジェクトを切り出して画像ファイルとして保存する。"""
    
    def __init__(
        self,
        output_dir: Path | str,
        margin: int = 10,
        image_format: str = "PNG"
    ) -> None:
        """初期化。
        
        Args:
            output_dir: 出力先ディレクトリ
            margin: 切り出し時に追加する余白（ピクセル）
            image_format: 画像フォーマット（"PNG", "JPEG"など）
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.margin = margin
        self.image_format = image_format
    
    def crop(
        self,
        image: bytes | Image.Image,
        figure: CaptionedFigure,
        page_number: int,
        figure_index: int
    ) -> FigureMetadata:
        """図を切り出して保存し、メタデータを返す。
        
        Args:
            image: 元のページ画像
            figure: キャプション付き図オブジェクト
            page_number: ページ番号
            figure_index: ページ内の図番号
            
        Returns:
            保存された図のメタデータ
        """
        # PIL Imageに変換
        if isinstance(image, bytes):
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image
        
        # バウンディングボックスに余白を追加
        bbox = figure.figure.bbox
        x0 = max(0, bbox.x0 - self.margin)
        y0 = max(0, bbox.y0 - self.margin)
        x1 = min(pil_image.width, bbox.x1 + self.margin)
        y1 = min(pil_image.height, bbox.y1 + self.margin)
        
        # 切り出し
        cropped = pil_image.crop((x0, y0, x1, y1))
        
        # ファイル名生成
        filename = f"page{page_number:03d}_fig{figure_index:02d}.{self.image_format.lower()}"
        output_path = self.output_dir / filename
        
        # 保存
        cropped.save(output_path, format=self.image_format)
        
        # メタデータ作成
        metadata = FigureMetadata(
            image_path=str(output_path),
            page_number=page_number,
            bbox={
                "x0": float(bbox.x0),
                "y0": float(bbox.y0),
                "x1": float(bbox.x1),
                "y1": float(bbox.y1)
            },
            caption=figure.caption,
            linked_text=figure.nearby_text,
            confidence=figure.figure.confidence
        )
        
        return metadata
    
    def save_meta(
        self,
        metadata_list: List[FigureMetadata],
        filename: str = "figures_metadata.json"
    ) -> Path:
        """メタデータをJSONファイルとして保存。
        
        Args:
            metadata_list: メタデータのリスト
            filename: 出力ファイル名
            
        Returns:
            保存されたファイルのパス
        """
        output_path = self.output_dir / filename
        
        data = [meta.to_dict() for meta in metadata_list]
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def extract_all(
        self,
        page_images: List[bytes | Image.Image],
        captioned_figures_per_page: List[List[CaptionedFigure]],
        start_page: int = 0
    ) -> List[FigureMetadata]:
        """複数ページから図を一括抽出。
        
        Args:
            page_images: ページ画像のリスト
            captioned_figures_per_page: 各ページのキャプション付き図オブジェクトのリスト
            start_page: 開始ページ番号
            
        Returns:
            全メタデータのリスト
        """
        all_metadata = []
        
        for page_idx, (image, figures) in enumerate(zip(page_images, captioned_figures_per_page)):
            page_number = start_page + page_idx
            
            for fig_idx, fig in enumerate(figures):
                metadata = self.crop(image, fig, page_number, fig_idx)
                all_metadata.append(metadata)
        
        # メタデータをJSON保存
        if all_metadata:
            self.save_meta(all_metadata)
        
        return all_metadata
