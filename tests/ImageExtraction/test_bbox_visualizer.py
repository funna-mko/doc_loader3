"""BBoxVisualizer のテスト。"""
from pathlib import Path

import pytest

from ImageExtraction.bbox_visualizer import BBoxVisualizer
from ImageExtraction.layout_analyzer import LayoutAnalyzer
from ImageExtraction.figure_clusterer import FigureClusterer
from ImageExtraction.pdf_renderer import PdfRenderer


@pytest.fixture()
def progit_pdf() -> Path:
    """progit.pdf ファイルのパスを返す。"""
    pdf_path = Path(__file__).parent.parent.parent / "data" / "progit.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDFファイルが見つかりません: {pdf_path}")
    return pdf_path


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """テスト出力用ディレクトリを作成して返す。"""
    output = tmp_path / "visualizations"
    output.mkdir(exist_ok=True)
    return output


def test_visualize_layout_elements(progit_pdf: Path, output_dir: Path) -> None:
    """progit.pdfの12-16ページ（13-17ページ目）のレイアウト要素を可視化。"""
    renderer = PdfRenderer(progit_pdf)
    analyzer = LayoutAnalyzer(min_area=2000)
    visualizer = BBoxVisualizer(line_width=2, show_labels=True)
    
    for page_num in range(12, 17):  # 12-16 (0始まりなので13-17ページ目)
        # ページをレンダリング
        png_bytes = renderer.render_page(page_num, zoom=1.5)
        
        # レイアウト要素を検出
        elements = analyzer.detect(png_bytes)
        
        # 可視化して保存
        output_path = output_dir / f"page_{page_num + 1:02d}_layout.png"
        visualizer.draw_layout_elements(png_bytes, elements, output_path)
        
        # ファイルが生成されたことを確認
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        print(f"Page {page_num + 1}: {len(elements)} elements detected -> {output_path}")


def test_visualize_figure_objects(progit_pdf: Path, output_dir: Path) -> None:
    """progit.pdfの12-16ページの図オブジェクトを可視化。"""
    renderer = PdfRenderer(progit_pdf)
    analyzer = LayoutAnalyzer(min_area=2000)
    clusterer = FigureClusterer(distance_threshold=100.0, overlap_threshold=0.3)
    visualizer = BBoxVisualizer(line_width=3, show_labels=True)
    
    for page_num in range(12, 17):  # 12-16
        # ページをレンダリング
        png_bytes = renderer.render_page(page_num, zoom=1.5)
        
        # レイアウト要素を検出してグルーピング
        elements = analyzer.detect(png_bytes)
        figures = clusterer.group(elements)
        
        # 可視化して保存
        output_path = output_dir / f"page_{page_num + 1:02d}_figures.png"
        visualizer.draw_figure_objects(png_bytes, figures, output_path)
        
        # ファイルが生成されたことを確認
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        print(f"Page {page_num + 1}: {len(figures)} figures grouped -> {output_path}")


def test_visualize_with_permanent_output(progit_pdf: Path) -> None:
    """永続的な出力ディレクトリに可視化結果を保存（手動確認用）。"""
    # プロジェクトルートの output ディレクトリに保存
    output_dir = Path(__file__).parent.parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    renderer = PdfRenderer(progit_pdf)
    analyzer = LayoutAnalyzer(min_area=2000)
    clusterer = FigureClusterer(distance_threshold=100.0, overlap_threshold=0.3)
    visualizer = BBoxVisualizer(line_width=3, show_labels=True)
    
    for page_num in range(12, 17):  # 12-16
        # ページをレンダリング
        png_bytes = renderer.render_page(page_num, zoom=1.5)
        
        # レイアウト要素を検出
        elements = analyzer.detect(png_bytes)
        output_layout = output_dir / f"page_{page_num + 1:02d}_layout.png"
        visualizer.draw_layout_elements(png_bytes, elements, output_layout)
        
        # 図オブジェクトをグルーピング
        figures = clusterer.group(elements)
        output_figures = output_dir / f"page_{page_num + 1:02d}_figures.png"
        visualizer.draw_figure_objects(png_bytes, figures, output_figures)
        
        print(f"Page {page_num + 1}: Saved to {output_dir}")
        print(f"  - {len(elements)} elements -> {output_layout.name}")
        print(f"  - {len(figures)} figures -> {output_figures.name}")
    
    print(f"\n可視化結果を確認してください: {output_dir.absolute()}")
