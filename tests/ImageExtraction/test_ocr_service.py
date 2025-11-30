"""OcrService のテスト。"""
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ImageExtraction.ocr_service import OcrService, OcrResult
from ImageExtraction.pdf_renderer import BBox


@pytest.fixture()
def tesseract_cmd() -> str:
    """Tesseractコマンドのパスを返す。"""
    # 一般的なWindowsのインストールパス
    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_path):
        return default_path
    
    # PATHから検索
    import shutil
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    
    return "tesseract"  # デフォルト


@pytest.fixture()
def text_image() -> Image.Image:
    """テキストを含むシンプルな画像を生成。"""
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    
    draw.text((20, 20), "Hello World", fill='black', font=font)
    draw.text((20, 100), "Test Image", fill='black', font=font)
    
    return img


def test_ocr_with_tesseract_if_available(text_image: Image.Image, tesseract_cmd: str) -> None:
    """pytesseract が利用可能ならOCRが動作することを確認。"""
    try:
        import pytesseract
    except ImportError:
        pytest.skip("pytesseract がインストールされていません")
    
    ocr = OcrService(engine="tesseract", lang="eng", tesseract_cmd=tesseract_cmd)
    results = ocr.run(text_image)
    
    # 何らかのテキストが検出されることを確認
    assert len(results) > 0
    
    # すべての結果が適切な型であることを確認
    for r in results:
        assert isinstance(r, OcrResult)
        assert isinstance(r.text, str)
        assert isinstance(r.bbox, BBox)
        assert 0.0 <= r.confidence <= 1.0
    
    # "Hello" か "World" か "Test" が含まれているか確認
    all_text = ' '.join(r.text for r in results)
    assert any(word in all_text for word in ['Hello', 'World', 'Test'])


def test_ocr_from_bytes(text_image: Image.Image, tesseract_cmd: str) -> None:
    """バイト列からOCRできることを確認。"""
    try:
        import pytesseract
    except ImportError:
        pytest.skip("pytesseract がインストールされていません")
    
    from io import BytesIO
    
    buf = BytesIO()
    text_image.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    
    ocr = OcrService(engine="tesseract", lang="eng", tesseract_cmd=tesseract_cmd)
    results = ocr.run(image_bytes)
    
    assert len(results) > 0


def test_extract_from_region(text_image: Image.Image, tesseract_cmd: str) -> None:
    """特定領域からテキストを抽出できることを確認。"""
    try:
        import pytesseract
    except ImportError:
        pytest.skip("pytesseract がインストールされていません")
    
    ocr = OcrService(engine="tesseract", lang="eng", tesseract_cmd=tesseract_cmd)
    
    # 上半分の領域を指定
    region = BBox(0.0, 0.0, 400.0, 100.0)
    text = ocr.extract_from_region(text_image, region)
    
    # 何らかのテキストが抽出されることを確認
    assert len(text) > 0


def test_unsupported_engine_raises() -> None:
    """未サポートのエンジンを指定すると例外が発生することを確認。"""
    ocr = OcrService(engine="unknown")
    
    with pytest.raises(ValueError, match="未サポートのOCRエンジン"):
        ocr.run(Image.new('RGB', (100, 100)))


@pytest.fixture()
def progit_pdf() -> Path:
    """progit.pdf ファイルのパスを返す。"""
    pdf_path = Path(__file__).parent.parent.parent / "data" / "progit.pdf"
    if not pdf_path.exists():
        pytest.skip(f"PDFファイルが見つかりません: {pdf_path}")
    return pdf_path


def test_ocr_on_pdf_page(progit_pdf: Path, tesseract_cmd: str) -> None:
    """実際のPDFページからOCRできることを確認。"""
    try:
        import pytesseract
    except ImportError:
        pytest.skip("pytesseract がインストールされていません")
    
    from ImageExtraction.pdf_renderer import PdfRenderer
    
    renderer = PdfRenderer(progit_pdf)
    png_bytes = renderer.render_page(12, zoom=1.5)
    
    ocr = OcrService(engine="tesseract", lang="jpn+eng", tesseract_cmd=tesseract_cmd)
    results = ocr.run(png_bytes)
    
    # 結果が得られることを確認
    print(f"\nOCR検出数: {len(results)}")
    
    if results:
        # 最初の数件を表示
        for i, r in enumerate(results[:5]):
            print(f"  {i}: '{r.text}' (conf={r.confidence:.2f})")
