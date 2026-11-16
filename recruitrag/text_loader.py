"""Load a resume file as text. PDF/DOCX go through Docling (-> markdown)."""
import os

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_converter = None


def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext in (".pdf", ".docx"):
        global _converter
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PDF/DOCX parsing needs Docling: pip install docling") from exc
        if _converter is None:
            _converter = DocumentConverter()
        return _converter.convert(path).document.export_to_markdown()
    raise ValueError(f"Unsupported file type: {ext}")
