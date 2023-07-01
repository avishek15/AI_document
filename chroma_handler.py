import os

import chromadb
from chromadb.config import Settings
from config import docs_cache, model_cache
from langchain.embeddings import SentenceTransformerEmbeddings
import pypdfium2 as pdfium

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2", cache_folder=model_cache)
chroma_client = chromadb.Client(Settings(
    persist_directory=docs_cache
))


def process_pdf(pdf_path):
    metadata = pdf_path.split(os.sep)[-1]
    doc = pdfium.PdfDocument(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc.get_page(page_num).get_textpage().get_text_range()

    print(len(full_text))
