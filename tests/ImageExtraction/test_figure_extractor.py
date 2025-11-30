"""FigureExtractor のテスト。"""
import json
from pathlib import Path

import pytest
from PIL import Image

from ImageExtraction.caption_linker import CaptionedFigure
from ImageExtraction.figure_clusterer import FigureObject
from ImageExtraction.figure_extractor import FigureExtractor, FigureMetadata
from ImageExtraction.layout_analyzer import LayoutElement, ElementType
from ImageExtraction.pdf_renderer import BBox


@pytest.fixture()
def sample_image() -> Image.Image:
    """テスト用のサンプル画像を生成。"""
    return Image.new('RGB', (800, 600), color='white')


@pytest.fixture()
def sample_captioned_figure() -> CaptionedFigure:
    """テスト用のキャプション付き図オブジェクトを生成。"""
    elem = LayoutElement(
        bbox=BBox(100.0, 100.0, 300.0, 300.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure = FigureObject(bbox=elem.bbox, elements=[elem], confidence=0.9)
    
    return CaptionedFigure(
        figure=figure,
        caption="図1: テスト図",
        nearby_text=["周辺テキスト1", "周辺テキスト2"]
    )


def test_crop_saves_image(
    tmp_path: Path,
    sample_image: Image.Image,
    sample_captioned_figure: CaptionedFigure
) -> None:
    """図を切り出して保存できることを確認。"""
    extractor = FigureExtractor(output_dir=tmp_path, margin=5)
    
    metadata = extractor.crop(sample_image, sample_captioned_figure, page_number=1, figure_index=0)
    
    # ファイルが生成されていることを確認
    assert Path(metadata.image_path).exists()
    assert Path(metadata.image_path).stat().st_size > 0
    
    # メタデータの内容を確認
    assert metadata.page_number == 1
    assert metadata.caption == "図1: テスト図"
    assert metadata.linked_text == ["周辺テキスト1", "周辺テキスト2"]
    assert metadata.confidence == 0.9


def test_crop_with_margin(
    tmp_path: Path,
    sample_image: Image.Image,
    sample_captioned_figure: CaptionedFigure
) -> None:
    """余白付きで切り出せることを確認。"""
    extractor = FigureExtractor(output_dir=tmp_path, margin=20)
    
    metadata = extractor.crop(sample_image, sample_captioned_figure, page_number=1, figure_index=0)
    
    # 切り出された画像を読み込んで確認
    cropped_img = Image.open(metadata.image_path)
    
    # 元のbboxは200x200だが、余白20が両側に追加されて240x240になる（または画像境界まで）
    assert cropped_img.width > 200
    assert cropped_img.height > 200


def test_save_meta(
    tmp_path: Path,
    sample_image: Image.Image,
    sample_captioned_figure: CaptionedFigure
) -> None:
    """メタデータをJSONとして保存できることを確認。"""
    extractor = FigureExtractor(output_dir=tmp_path)
    
    metadata1 = extractor.crop(sample_image, sample_captioned_figure, page_number=1, figure_index=0)
    metadata2 = extractor.crop(sample_image, sample_captioned_figure, page_number=1, figure_index=1)
    
    # JSONとして保存
    json_path = extractor.save_meta([metadata1, metadata2])
    
    # ファイルが生成されていることを確認
    assert json_path.exists()
    
    # JSON内容を確認
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert len(data) == 2
    assert data[0]['page_number'] == 1
    assert data[0]['caption'] == "図1: テスト図"


def test_extract_all(
    tmp_path: Path
) -> None:
    """複数ページから一括抽出できることを確認。"""
    extractor = FigureExtractor(output_dir=tmp_path)
    
    # ページ画像を生成
    page_images = [
        Image.new('RGB', (800, 600), color='white'),
        Image.new('RGB', (800, 600), color='lightgray')
    ]
    
    # 各ページの図オブジェクトを生成
    elem1 = LayoutElement(
        bbox=BBox(100.0, 100.0, 200.0, 200.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    fig1 = CaptionedFigure(
        figure=FigureObject(bbox=elem1.bbox, elements=[elem1], confidence=0.9),
        caption="Page 1 Fig"
    )
    
    elem2 = LayoutElement(
        bbox=BBox(300.0, 300.0, 400.0, 400.0),
        element_type=ElementType.FIGURE,
        confidence=0.8
    )
    fig2 = CaptionedFigure(
        figure=FigureObject(bbox=elem2.bbox, elements=[elem2], confidence=0.8),
        caption="Page 2 Fig"
    )
    
    captioned_figures_per_page = [[fig1], [fig2]]
    
    # 一括抽出
    all_metadata = extractor.extract_all(
        page_images,
        captioned_figures_per_page,
        start_page=0
    )
    
    # 結果を確認
    assert len(all_metadata) == 2
    assert all_metadata[0].page_number == 0
    assert all_metadata[0].caption == "Page 1 Fig"
    assert all_metadata[1].page_number == 1
    assert all_metadata[1].caption == "Page 2 Fig"
    
    # JSONファイルも自動保存されていることを確認
    json_path = tmp_path / "figures_metadata.json"
    assert json_path.exists()


def test_metadata_to_dict(
    tmp_path: Path,
    sample_image: Image.Image,
    sample_captioned_figure: CaptionedFigure
) -> None:
    """FigureMetadata が辞書に変換できることを確認。"""
    extractor = FigureExtractor(output_dir=tmp_path)
    
    metadata = extractor.crop(sample_image, sample_captioned_figure, page_number=1, figure_index=0)
    
    data = metadata.to_dict()
    
    assert isinstance(data, dict)
    assert 'image_path' in data
    assert 'page_number' in data
    assert 'bbox' in data
    assert 'caption' in data
    assert 'confidence' in data


@pytest.fixture()
def progit_pdf() -> Path:
    """progit.pdf ファイルのパスを返す。"""
    pdf_path = Path(__file__).parent.parent.parent / "data" / "progit.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDFファイルが見つかりません: {pdf_path}")
    return pdf_path


def test_extract_from_real_pdf(progit_pdf: Path, tmp_path: Path) -> None:
    """実際のPDFから図を抽出できることを確認。"""
    from ImageExtraction.pdf_renderer import PdfRenderer
    from ImageExtraction.layout_analyzer import LayoutAnalyzer
    from ImageExtraction.figure_clusterer import FigureClusterer
    from ImageExtraction.caption_linker import CaptionLinker
    
    renderer = PdfRenderer(progit_pdf)
    analyzer = LayoutAnalyzer(min_area=2000)
    clusterer = FigureClusterer(distance_threshold=100.0)
    linker = CaptionLinker(caption_distance_threshold=50.0)
    extractor = FigureExtractor(output_dir=tmp_path / "extracted")
    
    # 13ページ目を処理
    page_num = 12
    png_bytes = renderer.render_page(page_num, zoom=1.5)
    text_blocks = renderer.extract_text_blocks(page_num)
    
    # レイアウト解析 → クラスタリング → キャプション紐付け
    elements = analyzer.detect(png_bytes)
    figures = clusterer.group(elements)
    captioned_figures = linker.attach(figures, text_blocks)
    
    # 図を抽出
    if captioned_figures:
        metadata_list = []
        for i, fig in enumerate(captioned_figures):
            metadata = extractor.crop(png_bytes, fig, page_num, i)
            metadata_list.append(metadata)
        
        # JSONも保存
        extractor.save_meta(metadata_list)
        
        # 確認
        assert len(metadata_list) > 0
        print(f"\n{len(metadata_list)} 個の図を抽出しました: {tmp_path / 'extracted'}")
