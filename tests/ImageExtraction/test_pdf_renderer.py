"""Tests for PdfRenderer."""
from pathlib import Path
from typing import Iterable, List

import pytest

from ImageExtraction.pdf_renderer import BBox, PdfRenderer, TextBlock


class DummyPixmap:
    def __init__(self) -> None:
        self.formats: List[str] = []

    def tobytes(self, image_format: str) -> bytes:
        self.formats.append(image_format)
        return b"fake-bytes"


class DummyPage:
    def __init__(self, blocks: Iterable[tuple]) -> None:
        self.blocks = list(blocks)
        self.matrices: List[object] = []

    def get_pixmap(self, matrix: object) -> DummyPixmap:
        self.matrices.append(matrix)
        return DummyPixmap()

    def get_text(self, option: str) -> Iterable[tuple]:
        assert option == "blocks"
        return self.blocks


class DummyDoc:
    def __init__(self, blocks: Iterable[tuple]) -> None:
        self.page_count = 1
        self.blocks = blocks
        self.loaded_pages: List[int] = []
        self.closed = False

    def load_page(self, page_number: int) -> DummyPage:
        self.loaded_pages.append(page_number)
        return DummyPage(self.blocks)

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def existing_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    return pdf_path


def test_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        PdfRenderer(missing)


def test_render_page_returns_png_bytes(existing_pdf: Path) -> None:
    doc = DummyDoc(blocks=[])
    renderer = PdfRenderer(existing_pdf, doc_factory=lambda _: doc)

    result = renderer.render_page(0)

    assert result == b"fake-bytes"
    assert doc.loaded_pages == [0]


def test_extract_text_blocks_filters_empty(existing_pdf: Path) -> None:
    blocks = [
        (0, 0, 10, 10, "Title"),
        (0, 10, 10, 20, ""),
        (0, 20, 10, 30, None),
    ]
    doc = DummyDoc(blocks=blocks)
    renderer = PdfRenderer(existing_pdf, doc_factory=lambda _: doc)

    results = renderer.extract_text_blocks(0)

    assert results == [TextBlock(text="Title", bbox=BBox(0.0, 0.0, 10.0, 10.0))]


@pytest.fixture()
def progit_pdf() -> Path:
    """progit.pdf ファイルのパスを返す。"""
    pdf_path = Path(__file__).parent.parent.parent / "data" / "progit.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDFファイルが見つかりません: {pdf_path}")
    return pdf_path


def test_render_progit_pages(progit_pdf: Path) -> None:
    """progit.pdfの13-15ページをレンダリングしてPNG画像が取得できることを確認する。"""
    renderer = PdfRenderer(progit_pdf)
    
    for page_num in range(12, 15):  # 13-15ページ（0始まりなので12-14）
        png_bytes = renderer.render_page(page_num, zoom=1.5)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes.startswith(b'\x89PNG')  # PNG形式のマジックナンバー


def test_extract_text_from_progit_pages(progit_pdf: Path) -> None:
    """progit.pdfの13-15ページからテキストブロックを抽出する。"""
    renderer = PdfRenderer(progit_pdf)
    
    for page_num in range(12, 15):  # 13-15ページ（0始まりなので12-14）
        text_blocks = renderer.extract_text_blocks(page_num)
        assert isinstance(text_blocks, list)
        assert len(text_blocks) > 0  # 何らかのテキストが存在するはず
        
        # 各テキストブロックの構造を確認
        for block in text_blocks:
            assert isinstance(block, TextBlock)
            assert isinstance(block.text, str)
            assert len(block.text) > 0
            assert isinstance(block.bbox, BBox)
            assert block.bbox.x1 > block.bbox.x0
            assert block.bbox.y1 > block.bbox.y0
            print(f"Page {page_num + 1} Block: {block.text[:30]}... BBox: {block.bbox}")
