"""FigureClusterer のテスト。"""
import pytest

from ImageExtraction.figure_clusterer import FigureClusterer, FigureObject
from ImageExtraction.layout_analyzer import ElementType, LayoutElement
from ImageExtraction.pdf_renderer import BBox


def test_group_empty_list() -> None:
    """空のリストを渡した場合、空のリストが返ることを確認。"""
    clusterer = FigureClusterer()
    
    result = clusterer.group([])
    
    assert result == []


def test_group_single_element() -> None:
    """単一要素の場合、そのまま1つの図オブジェクトになることを確認。"""
    clusterer = FigureClusterer()
    
    elements = [
        LayoutElement(
            bbox=BBox(10.0, 10.0, 100.0, 100.0),
            element_type=ElementType.FIGURE,
            confidence=0.9
        )
    ]
    
    result = clusterer.group(elements)
    
    assert len(result) == 1
    assert result[0].bbox == BBox(10.0, 10.0, 100.0, 100.0)
    assert len(result[0].elements) == 1


def test_group_distant_elements() -> None:
    """離れた要素は別々の図オブジェクトになることを確認。"""
    clusterer = FigureClusterer(distance_threshold=50.0)
    
    elements = [
        LayoutElement(
            bbox=BBox(10.0, 10.0, 50.0, 50.0),
            element_type=ElementType.FIGURE,
            confidence=0.8
        ),
        LayoutElement(
            bbox=BBox(200.0, 200.0, 250.0, 250.0),
            element_type=ElementType.FIGURE,
            confidence=0.8
        )
    ]
    
    result = clusterer.group(elements)
    
    assert len(result) == 2


def test_group_nearby_elements() -> None:
    """近接した要素は統合されることを確認。"""
    clusterer = FigureClusterer(distance_threshold=100.0)
    
    elements = [
        LayoutElement(
            bbox=BBox(10.0, 10.0, 50.0, 50.0),
            element_type=ElementType.FIGURE,
            confidence=0.8
        ),
        LayoutElement(
            bbox=BBox(60.0, 10.0, 100.0, 50.0),  # 横に近接
            element_type=ElementType.FIGURE,
            confidence=0.9
        )
    ]
    
    result = clusterer.group(elements)
    
    assert len(result) == 1
    # 統合されたバウンディングボックスは両方を含む
    assert result[0].bbox.x0 == 10.0
    assert result[0].bbox.x1 == 100.0
    assert len(result[0].elements) == 2


def test_group_overlapping_elements() -> None:
    """重なり合う要素は統合されることを確認。"""
    clusterer = FigureClusterer(overlap_threshold=0.2)
    
    elements = [
        LayoutElement(
            bbox=BBox(10.0, 10.0, 100.0, 100.0),
            element_type=ElementType.FIGURE,
            confidence=0.8
        ),
        LayoutElement(
            bbox=BBox(50.0, 50.0, 150.0, 150.0),  # 部分的に重なる
            element_type=ElementType.FIGURE,
            confidence=0.9
        )
    ]
    
    result = clusterer.group(elements)
    
    assert len(result) == 1
    assert result[0].bbox.x0 == 10.0
    assert result[0].bbox.y0 == 10.0
    assert result[0].bbox.x1 == 150.0
    assert result[0].bbox.y1 == 150.0


def test_group_multiple_clusters() -> None:
    """複数のクラスタが正しく形成されることを確認。"""
    clusterer = FigureClusterer(distance_threshold=50.0)
    
    elements = [
        # クラスタ1
        LayoutElement(
            bbox=BBox(10.0, 10.0, 30.0, 30.0),
            element_type=ElementType.FIGURE,
            confidence=0.8
        ),
        LayoutElement(
            bbox=BBox(35.0, 10.0, 55.0, 30.0),
            element_type=ElementType.FIGURE,
            confidence=0.8
        ),
        # クラスタ2
        LayoutElement(
            bbox=BBox(200.0, 200.0, 220.0, 220.0),
            element_type=ElementType.FIGURE,
            confidence=0.9
        ),
        LayoutElement(
            bbox=BBox(225.0, 200.0, 245.0, 220.0),
            element_type=ElementType.FIGURE,
            confidence=0.9
        )
    ]
    
    result = clusterer.group(elements)
    
    assert len(result) == 2
    # 各クラスタは2つの要素を含む
    assert all(len(fig.elements) == 2 for fig in result)


def test_calculate_overlap() -> None:
    """重なり率計算が正しく動作することを確認。"""
    clusterer = FigureClusterer()
    
    # 完全一致
    bbox1 = BBox(0.0, 0.0, 10.0, 10.0)
    bbox2 = BBox(0.0, 0.0, 10.0, 10.0)
    assert clusterer._calculate_overlap(bbox1, bbox2) == 1.0
    
    # 重なりなし
    bbox3 = BBox(0.0, 0.0, 10.0, 10.0)
    bbox4 = BBox(20.0, 20.0, 30.0, 30.0)
    assert clusterer._calculate_overlap(bbox3, bbox4) == 0.0
    
    # 50%重なり
    bbox5 = BBox(0.0, 0.0, 10.0, 10.0)
    bbox6 = BBox(5.0, 0.0, 15.0, 10.0)
    overlap = clusterer._calculate_overlap(bbox5, bbox6)
    assert 0.4 < overlap < 0.6  # 約0.5


def test_calculate_distance() -> None:
    """距離計算が正しく動作することを確認。"""
    clusterer = FigureClusterer()
    
    # 同じ中心
    bbox1 = BBox(0.0, 0.0, 10.0, 10.0)
    bbox2 = BBox(0.0, 0.0, 10.0, 10.0)
    assert clusterer._calculate_distance(bbox1, bbox2) == 0.0
    
    # 横に10離れた位置
    bbox3 = BBox(0.0, 0.0, 10.0, 10.0)  # 中心 (5, 5)
    bbox4 = BBox(10.0, 0.0, 20.0, 10.0)  # 中心 (15, 5)
    distance = clusterer._calculate_distance(bbox3, bbox4)
    assert abs(distance - 10.0) < 0.01


def test_merge_figure_objects() -> None:
    """FigureObject のマージが正しく動作することを確認。"""
    elem1 = LayoutElement(
        bbox=BBox(10.0, 10.0, 50.0, 50.0),
        element_type=ElementType.FIGURE,
        confidence=0.8
    )
    elem2 = LayoutElement(
        bbox=BBox(60.0, 60.0, 100.0, 100.0),
        element_type=ElementType.FIGURE,
        confidence=0.6
    )
    
    fig1 = FigureObject(bbox=elem1.bbox, elements=[elem1], confidence=0.8)
    fig2 = FigureObject(bbox=elem2.bbox, elements=[elem2], confidence=0.6)
    
    merged = fig1.merge(fig2)
    
    # バウンディングボックスは両方を含む
    assert merged.bbox.x0 == 10.0
    assert merged.bbox.y0 == 10.0
    assert merged.bbox.x1 == 100.0
    assert merged.bbox.y1 == 100.0
    
    # 要素は両方含まれる
    assert len(merged.elements) == 2
    
    # 信頼度は平均
    assert merged.confidence == 0.7
