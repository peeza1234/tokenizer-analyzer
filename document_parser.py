import json, os, io
import pdfplumber, fitz, pytesseract
from PIL import Image
 
AZURE_ENDPOINT = os.environ.get("AZURE_DOCINTEL_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_DOCINTEL_KEY")
 
 
def table_to_markdown(rows):
    rows = [[(c or "").strip() for c in row] for row in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)
 
 
def extract_native_page(page):
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words:
        return [], []
 
    mid = page.width / 2
    left = [w for w in words if w["x0"] < mid]
    right = [w for w in words if w["x0"] >= mid]
    groups = [left, right] if len(left) > 3 and len(right) > 3 else [words]
 
    text_blocks = []
    for group in groups:
        lines = {}
        for w in group:
            lines.setdefault(round(w["top"]), []).append(w)
        for top in sorted(lines):
            row = sorted(lines[top], key=lambda w: w["x0"])
            text_blocks.append(" ".join(w["text"] for w in row))
 
    tables = [table_to_markdown(t) for t in page.extract_tables() if t]
    return text_blocks, tables
 
 
def render_page_as_image(fitz_page, zoom=2.0):
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    return Image.open(io.BytesIO(pix.tobytes("png")))
 
 
def ocr_with_azure(img):
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
 
    client = DocumentIntelligenceClient(AZURE_ENDPOINT, AzureKeyCredential(AZURE_KEY))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    result = client.begin_analyze_document("prebuilt-layout", buf).result()
 
    blocks = [l.content for p in result.pages for l in (p.lines or [])]
    tables = []
    for t in result.tables or []:
        grid = [["" for _ in range(t.column_count)] for _ in range(t.row_count)]
        for cell in t.cells:
            grid[cell.row_index][cell.column_index] = cell.content
        tables.append(table_to_markdown(grid))
    return blocks, tables
 
 
def run_ocr_fallback(fitz_page):
    img = render_page_as_image(fitz_page)
    if AZURE_ENDPOINT and AZURE_KEY:
        try:
            blocks, tables = ocr_with_azure(img)
            return blocks, tables, "ocr_azure"
        except Exception as e:
            print(f"(Azure failed -- {e}. Falling back to Tesseract.)")
    text = pytesseract.image_to_string(img)
    blocks = [l.strip() for l in text.split("\n") if l.strip()]
    return blocks, [], "ocr_tesseract"
 
 
def parse_document(pdf_path):
    fitz_doc = fitz.open(pdf_path)
    pages = []
    with pdfplumber.open(pdf_path) as doc:
        for i, page in enumerate(doc.pages):
            blocks, tables = extract_native_page(page)
            source = "native"
            if not blocks and not tables:
                blocks, tables, source = run_ocr_fallback(fitz_doc[i])
            pages.append({"page_number": i + 1, "text_blocks": blocks, "tables_markdown": tables, "source": source})
    fitz_doc.close()
    return {"pages": pages}
 
 
def run_tests(result):
    required = {"page_number", "text_blocks", "tables_markdown", "source"}
    for p in result["pages"]:
        assert set(p.keys()) == required
        assert isinstance(p["text_blocks"], list) and isinstance(p["tables_markdown"], list)
        assert p["source"] in ("native", "ocr_tesseract", "ocr_azure")
        json.dumps(p)
    print(f"All checks passed for {len(result['pages'])} pages.")
 
 
if __name__ == "__main__":
    pdf_path = os.environ.get("PDF_PATH", "corporate_report.pdf")
    result = parse_document(pdf_path)
    run_tests(result)
    with open("parsed_output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Written to parsed_output.json")