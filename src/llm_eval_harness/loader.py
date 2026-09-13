from pypdf import PdfReader


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text())
    return "\n".join(pages)
