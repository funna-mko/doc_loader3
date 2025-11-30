"""CaptionLinker のテスト。"""
import pytest

from ImageExtraction.caption_linker import CaptionLinker, CaptionedFigure
from ImageExtraction.figure_clusterer import FigureObject
from ImageExtraction.layout_analyzer import LayoutElement, ElementType
from ImageExtraction.pdf_renderer import BBox, TextBlock


def test_attach_caption_below_figure() -> None:
    """図の直下にあるテキストがキャプションとして紐付けられることを確認。"""
    linker = CaptionLinker(caption_distance_threshold=50.0)
    
    # 図オブジェクト
    elem = LayoutElement(
        bbox=BBox(100.0, 100.0, 300.0, 200.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure = FigureObject(bbox=elem.bbox, elements=[elem], confidence=0.9)
    
    # テキストブロック（図の直下）
    text_blocks = [
        TextBlock(
            text="図1: サンプル図",
            bbox=BBox(100.0, 210.0, 300.0, 230.0)  # 図の少し下
        )
    ]
    
    results = linker.attach([figure], text_blocks)
    
    assert len(results) == 1
    assert results[0].caption == "図1: サンプル図"


def test_attach_caption_above_figure() -> None:
    """図の直上にあるテキストもキャプションとして紐付けられることを確認。"""
    linker = CaptionLinker(caption_distance_threshold=50.0)
    
    elem = LayoutElement(
        bbox=BBox(100.0, 100.0, 300.0, 200.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure = FigureObject(bbox=elem.bbox, elements=[elem], confidence=0.9)
    
    # テキストブロック（図の直上）
    text_blocks = [
        TextBlock(
            text="Figure 1",
            bbox=BBox(100.0, 70.0, 300.0, 90.0)  # 図の少し上
        )
    ]
    
    results = linker.attach([figure], text_blocks)
    
    assert len(results) == 1
    assert results[0].caption == "Figure 1"


def test_attach_no_caption() -> None:
    """近くにテキストがない場合、キャプションがNoneになることを確認。"""
    linker = CaptionLinker(caption_distance_threshold=50.0)
    
    elem = LayoutElement(
        bbox=BBox(100.0, 100.0, 300.0, 200.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure = FigureObject(bbox=elem.bbox, elements=[elem], confidence=0.9)
    
    # 遠くにあるテキスト
    text_blocks = [
        TextBlock(
            text="遠いテキスト",
            bbox=BBox(500.0, 500.0, 600.0, 520.0)
        )
    ]
    
    results = linker.attach([figure], text_blocks)
    
    assert len(results) == 1
    assert results[0].caption is None


def test_attach_nearby_text() -> None:
    """周辺テキストが収集されることを確認。"""
    linker = CaptionLinker(
        caption_distance_threshold=50.0,
        context_distance_threshold=200.0  # 距離閾値を拡大
    )
    
    elem = LayoutElement(
        bbox=BBox(100.0, 100.0, 300.0, 200.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure = FigureObject(bbox=elem.bbox, elements=[elem], confidence=0.9)
    
    text_blocks = [
        TextBlock(text="近いテキスト1", bbox=BBox(100.0, 210.0, 300.0, 230.0)),
        TextBlock(text="近いテキスト2", bbox=BBox(310.0, 100.0, 400.0, 120.0)),
        TextBlock(text="遠いテキスト", bbox=BBox(500.0, 500.0, 600.0, 520.0))
    ]
    
    results = linker.attach([figure], text_blocks)
    
    assert len(results) == 1
    assert len(results[0].nearby_text) >= 2
    assert "近いテキスト1" in results[0].nearby_text
    assert "近いテキスト2" in results[0].nearby_text


def test_fetch_context() -> None:
    """fetch_context が距離順でテキストを返すことを確認。"""
    linker = CaptionLinker()
    
    elem = LayoutElement(
        bbox=BBox(100.0, 100.0, 200.0, 200.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure = FigureObject(bbox=elem.bbox, elements=[elem], confidence=0.9)
    
    text_blocks = [
        TextBlock(text="遠い", bbox=BBox(300.0, 300.0, 400.0, 320.0)),
        TextBlock(text="近い", bbox=BBox(210.0, 100.0, 250.0, 120.0)),
        TextBlock(text="中間", bbox=BBox(250.0, 250.0, 300.0, 270.0))
    ]
    
    context = linker.fetch_context(figure, text_blocks, max_distance=200.0)
    
    # 距離順にソートされているはず
    assert len(context) >= 2
    assert context[0].text == "近い"  # 最も近い


def test_attach_multiple_figures() -> None:
    """複数の図にそれぞれテキストが紐付けられることを確認。"""
    linker = CaptionLinker(caption_distance_threshold=50.0)
    
    # 図1
    elem1 = LayoutElement(
        bbox=BBox(100.0, 100.0, 200.0, 200.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure1 = FigureObject(bbox=elem1.bbox, elements=[elem1], confidence=0.9)
    
    # 図2
    elem2 = LayoutElement(
        bbox=BBox(300.0, 300.0, 400.0, 400.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure2 = FigureObject(bbox=elem2.bbox, elements=[elem2], confidence=0.9)
    
    text_blocks = [
        TextBlock(text="図1のキャプション", bbox=BBox(100.0, 210.0, 200.0, 230.0)),
        TextBlock(text="図2のキャプション", bbox=BBox(300.0, 410.0, 400.0, 430.0))
    ]
    
    results = linker.attach([figure1, figure2], text_blocks)
    
    assert len(results) == 2
    assert results[0].caption == "図1のキャプション"
    assert results[1].caption == "図2のキャプション"


def test_captioned_figure_default_nearby_text() -> None:
    """CaptionedFigure の nearby_text がデフォルトで空リストになることを確認。"""
    elem = LayoutElement(
        bbox=BBox(0.0, 0.0, 10.0, 10.0),
        element_type=ElementType.FIGURE,
        confidence=0.9
    )
    figure = FigureObject(bbox=elem.bbox, elements=[elem], confidence=0.9)
    
    captioned = CaptionedFigure(figure=figure)
    
    assert captioned.nearby_text == []
