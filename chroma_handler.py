import os
import numpy as np
import chromadb
from chromadb.config import Settings
from config import docs_cache, model_cache, collection_name, context_len, overlap
from langchain.embeddings import SentenceTransformerEmbeddings
import pypdfium2 as pdfium

embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-MiniLM-L6-v2", cache_folder=model_cache)

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=docs_cache
))


def process_pdf(pdf_path):
    metadata = pdf_path.split(os.sep)[-1]
    doc = pdfium.PdfDocument(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc.get_page(page_num).get_textpage().get_text_range()
    doc.close()
    collection = chroma_client.get_or_create_collection(name=collection_name)
    id_offset = collection.count()
    doc_collection = []
    metadata_collection = []
    ids_collection = []
    for idx, i in enumerate(range(0, len(full_text) - context_len, context_len - overlap)):
        doc_collection.append(' '.join(full_text[i: i+context_len].split(' ')[1:-1]))
        metadata_collection.append({'book': metadata, 'chunk': idx})
        ids_collection.append(str(id_offset + idx))
    i = i + (context_len - overlap)
    doc_collection.append(full_text[i:])
    metadata_collection.append({'book': metadata, 'chunk': idx + 1})
    ids_collection.append(str(id_offset + idx + 1))
    random_ids_to_match = np.random.default_rng().choice(len(doc_collection), size=10, replace=False)
    if id_offset:
        matches = [collection.query(query_texts=doc_collection[random_id], n_results=1)['distances'][0] for random_id in
                   random_ids_to_match]
        matches = [match[0] for match in matches if len(match)]
        match_distance = np.sum(matches)
        if not match_distance:
            return
    collection.add(documents=doc_collection,
                   metadatas=metadata_collection,
                   ids=ids_collection)
    chroma_client.persist()
    return


def process_query(query):
    collection = chroma_client.get_collection(name=collection_name)
    query_result = collection.query(query_texts=query,
                                    n_results=3,
                                    include=["metadatas",
                                             "documents",
                                             "distances"]
                                    )

    return query_result
