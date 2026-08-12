import json
import fitz
import spacy
from nltk.stem import PorterStemmer, WordNetLemmatizer

PDF_PATH = "sample.pdf"
TARGET_LABELS = {"ORG", "DATE", "MONEY"}
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def demo_stem_vs_lemma(words):
    results = []
    for word in words:
        results.append({"word": word, "stem": stemmer.stem(word), "lemma": lemmatizer.lemmatize(word, pos="v")})
    return results

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    metadata = {"document_title": doc.metadata.get("title") or pdf_path, "total_pages": doc.page_count}
    pages = []
    for page_number in range(doc.page_count):
        page = doc.load_page(page_number)
        pages.append({"page_number": page_number + 1, "text": page.get_text()})
    doc.close()
    return metadata, pages

def extract_entities(pages):
    report = []
    for page in pages:
        doc = nlp(page["text"])
        for ent in doc.ents:
            if ent.label_ in TARGET_LABELS:
                report.append({"page_number": page["page_number"], "entity_text": ent.text, "entity_label": ent.label_, "lemma": ent.lemma_})
    return report

def build_report(pdf_path):
    metadata, pages = extract_pages(pdf_path)
    entities = extract_entities(pages)
    return {"document_metadata": metadata, "entities": entities}

if __name__ == "__main__":
    sample_words = ["running", "studies", "better", "companies", "organized"]
    print(json.dumps({"stemming_vs_lemmatization": demo_stem_vs_lemma(sample_words)}, indent=2))
    print(json.dumps(build_report(PDF_PATH), indent=2))