"""LayoutAnalyzer のテスト。"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ImageExtraction.layout_analyzer import (
    ElementType,
    LayoutAnalyzer,
    LayoutElement,
)
from ImageExtraction.pdf_renderer import BBox, PdfRenderer


@pytest.fixture()
def simple_image() -> Image.Image:
    """テスト用のシンプルな画像を生成。"""
    # 白背景に黒い矩形を3つ配置
    img = Image.new('L', (800, 600), color=255)  # 白背景
    pixels = np.array(img)
    
    # 矩形1: 横長（テキスト的）
    pixels[50:70, 50:300] = 0
    
    # 矩形2: 正方形（図的）
    pixels[100:250, 50:200] = 0
    
    # 矩形3: 縦長（表的）
    pixels[300:500, 50:150] = 0
    
    return Image.fromarray(pixels)


def test_detect_returns_elements(simple_image: Image.Image) -> None:
    """レイアウト要素が検出されることを確認。"""
    analyzer = LayoutAnalyzer(min_area=500)
    
    elements = analyzer.detect(simple_image)
    
    assert len(elements) > 0
    for elem in elements:
        assert isinstance(elem, LayoutElement)
        assert isinstance(elem.bbox, BBox)
        assert isinstance(elem.element_type, ElementType)
        assert 0.0 <= elem.confidence <= 1.0


def test_detect_filters_small_elements(simple_image: Image.Image) -> None:
    """小さい要素がフィルタされることを確認。"""
    analyzer_small = LayoutAnalyzer(min_area=100)
    analyzer_large = LayoutAnalyzer(min_area=10000)
    
    elements_small = analyzer_small.detect(simple_image)
    elements_large = analyzer_large.detect(simple_image)
    
    assert len(elements_small) >= len(elements_large)


def test_detect_from_bytes(simple_image: Image.Image) -> None:
    """バイト列からも検出できることを確認。"""
    from io import BytesIO
    
    buf = BytesIO()
    simple_image.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    
    analyzer = LayoutAnalyzer(min_area=500)
    elements = analyzer.detect(image_bytes)
    
    assert len(elements) > 0


def test_align_text() -> None:
    """テキストブロックとの紐付けが動作することを確認。"""
    analyzer = LayoutAnalyzer()
    
    elements = [
        LayoutElement(
            bbox=BBox(10.0, 10.0, 100.0, 50.0),
            element_type=ElementType.TEXT,
            confidence=0.8
        ),
        LayoutElement(
            bbox=BBox(200.0, 200.0, 300.0, 300.0),
            element_type=ElementType.FIGURE,
            confidence=0.9
        )
    ]
    
    text_blocks = [
        ("Hello World", BBox(10.0, 10.0, 100.0, 50.0)),  # 完全一致
        ("Another Text", BBox(150.0, 150.0, 250.0, 250.0))  # 部分的に重なる
    ]
    
    results = analyzer.align_text(elements, text_blocks)
    
    assert len(results) == 2
    assert results[0][1] == "Hello World"
    assert results[1][1] == "Another Text"


@pytest.fixture()
def progit_pdf() -> Path:
    """progit.pdf ファイルのパスを返す。"""
    pdf_path = Path(__file__).parent.parent.parent / "data" / "progit.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDFファイルが見つかりません: {pdf_path}")
    return pdf_path


def test_detect_from_pdf_page(progit_pdf: Path) -> None:
    """実際のPDFページから要素を検出できることを確認。"""
    renderer = PdfRenderer(progit_pdf)
    png_bytes = renderer.render_page(12, zoom=1.5)  # 13ページ目
    
    analyzer = LayoutAnalyzer(min_area=2000)
    elements = analyzer.detect(png_bytes)
    
    assert len(elements) > 0
    # 要素の詳細を出力（デバッグ用）
    for i, elem in enumerate(elements[:5]):  # 最初の5要素のみ
        print(f"Element {i}: type={elem.element_type.value}, "
              f"bbox=({elem.bbox.x0:.0f}, {elem.bbox.y0:.0f}, "
              f"{elem.bbox.x1:.0f}, {elem.bbox.y1:.0f}), "
              f"confidence={elem.confidence:.2f}")
