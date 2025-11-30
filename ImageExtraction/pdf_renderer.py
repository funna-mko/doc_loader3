"""PDFページのレンダリングとテキストメタデータの抽出を行うユーティリティ。"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol

try:  # PyMuPDFが利用可能な場合に読み込む。
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - PyMuPDFが存在しない場合のみ実行される。
    fitz = None  # type: ignore


# 以下のProtocolクラスは型チェック専用の構造的部分型を定義する。
# ここでは実装せず、実行時はPyMuPDF (fitz) の実際のオブジェクト、
# テスト時はダミーオブジェクト (DummyDoc/DummyPage/DummyPixmap) が実装を提供する。


class _PixmapProtocol(Protocol):
    """PdfRendererが必要とするPyMuPDF Pixmap APIのサブセット。"""

    def tobytes(self, image_format: str) -> bytes:
        """指定された形式でバイナリ画像データを返す。"""
        ...


class _PageProtocol(Protocol):
    """PdfRendererが必要とするPyMuPDF Page APIのサブセット。"""

    def get_pixmap(self, matrix: object) -> _PixmapProtocol:
        ...

    def get_text(self, option: str) -> Iterable[tuple]:
        ...


class _DocumentProtocol(Protocol):
    """PdfRendererが必要とするPyMuPDF Document APIのサブセット。"""

    page_count: int

    def load_page(self, page_number: int) -> _PageProtocol:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class BBox:
    """PDF座標空間におけるバウンディングボックス。"""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class TextBlock:
    """テキスト内容とそのバウンディングボックス。"""

    text: str
    bbox: BBox


class PdfRenderer:
    """PDFページを画像にレンダリングし、テキストブロックを列挙する。"""

    def __init__(
        self,
        pdf_path: Path | str,
        *,
        doc_factory: Optional[Callable[[Path], _DocumentProtocol]] = None,
    ) -> None:
        self._path = Path(pdf_path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)

        self._doc_factory = doc_factory or self._default_doc_factory
        self._doc: Optional[_DocumentProtocol] = None

    def _ensure_document(self) -> _DocumentProtocol:
        if self._doc is None:
            self._doc = self._doc_factory(self._path)
        return self._doc

    @staticmethod
    def _default_doc_factory(path: Path) -> _DocumentProtocol:
        if fitz is None:  # pragma: no cover - 本番環境では実際のインポートが必要。
            raise ImportError(
                "PdfRendererにはPyMuPDF (fitz) が必要です。カスタムdoc_factoryを指定しない限り必須です。"
            )
        return fitz.open(path)  # type: ignore[no-any-return]

    def _get_page(self, page_number: int) -> _PageProtocol:
        doc = self._ensure_document()
        if page_number < 0 or page_number >= doc.page_count:
            raise IndexError(f"ページ番号 {page_number} が範囲外です (0-{doc.page_count - 1})。")
        return doc.load_page(page_number)

    @staticmethod
    def _make_matrix(zoom: float) -> object:
        if fitz is None:
            return _ZoomMatrix(zoom, zoom)
        return fitz.Matrix(zoom, zoom)  # type: ignore[no-any-return]

    def render_page(self, page_number: int, *, zoom: float = 2.0) -> bytes:
        """指定されたページのPNGレンダリング結果を生バイト列として返す。"""

        page = self._get_page(page_number)
        pixmap = page.get_pixmap(matrix=self._make_matrix(zoom))
        return pixmap.tobytes("png")

    def extract_text_blocks(self, page_number: int) -> List[TextBlock]:
        """指定されたページのバウンディングボックスとテキストを抽出する。"""

        page = self._get_page(page_number)
        blocks = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            text = (block[4] or "").strip()
            if not text:
                continue
            bbox = BBox(x0=float(block[0]), y0=float(block[1]), x1=float(block[2]), y1=float(block[3]))
            blocks.append(TextBlock(text=text, bbox=bbox))
        return blocks

    def close(self) -> None:
        doc = self._doc
        if doc is not None:
            doc.close()
            self._doc = None

    def __enter__(self) -> "PdfRenderer":
        self._ensure_document()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class _ZoomMatrix:
    """PyMuPDFが利用できない場合のフォールバック行列（テスト専用）。"""

    x_scale: float
    y_scale: float

    def __iter__(self):  # pragma: no cover - 診断用ヘルパー。
        yield self.x_scale
        yield self.y_scale
