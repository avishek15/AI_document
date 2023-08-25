import os
import numpy as np
import chromadb
from chromadb.config import Settings
from config import docs_cache, collection_name, context_len, overlap, book_collection_name
from embeddings import bge_base_embeddings as embeddings
import pypdfium2 as pdfium

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=docs_cache
))


def process_pdf(pdf_path):
    metadata = pdf_path.split(os.sep)[-1][:-4]
    doc = pdfium.PdfDocument(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc.get_page(page_num).get_textpage().get_text_range()
    doc.close()
    collection = chroma_client.get_or_create_collection(name=collection_name,
                                                        embedding_function=embeddings.embed_documents,
                                                        metadata={"hnsw:space": "ip"})
    book_collection = chroma_client.get_or_create_collection(name=book_collection_name,
                                                             embedding_function=embeddings.embed_documents,
                                                             metadata={"hnsw:space": "ip"})
    id_offset = collection.count()
    doc_collection = []
    metadata_collection = []
    ids_collection = []
    for idx, i in enumerate(range(0, len(full_text) - context_len, context_len - overlap)):
        doc_collection.append(' '.join(full_text[i: i + context_len].split(' ')[1:-1]))
        metadata_collection.append({'book': metadata, 'chunk': idx})
        ids_collection.append(str(id_offset + idx))
    i = i + (context_len - overlap)
    doc_collection.append(full_text[i:])
    metadata_collection.append({'book': metadata, 'chunk': idx + 1})
    ids_collection.append(str(id_offset + idx + 1))
    random_ids_to_match = np.random.default_rng().choice(len(doc_collection), size=min(10, len(doc_collection)),
                                                         replace=False)
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
    book_collection.add(documents=[metadata],
                        ids=[str(book_collection.count())])
    chroma_client.persist()
    return


def process_query(query):
    num_records = 3
    try:
        collection = chroma_client.get_collection(name=collection_name,
                                                  embedding_function=embeddings.embed_documents)
        query_result = collection.query(query_texts=query,
                                        n_results=num_records,
                                        include=["metadatas",
                                                 "documents",
                                                 "distances"]
                                        )

        return query_result
    except ValueError as ve:
        return [""] * num_records


def get_top_page(query):
    try:
        book_name, search_query = query.strip().split(":")
        collection = chroma_client.get_collection(name=collection_name,
                                                  embedding_function=embeddings.embed_documents)
        book_collection = chroma_client.get_collection(name=book_collection_name,
                                                       embedding_function=embeddings.embed_documents)

        book_search = book_collection.query(query_texts=book_name, n_results=1,
                                               include=["documents"])
        potential_book = book_search["documents"][0][0]
        # if potential_book empty => handle
        # where_clause = {
        #     "$and": [
        #         {"book": {"$eq": potential_book}},
        #         {"chunk": {"$eq": int(search_query)}}
        #     ]
        # }
        where_clause = {"book": potential_book}
        query_result = collection.query(query_texts=search_query,
                                        n_results=1,
                                        include=["metadatas",
                                                 "documents",
                                                 "distances"],
                                        where=where_clause
                                        )

        return query_result["documents"][0][0]
    except ValueError as ve:
        print(ve)
        return "No data found!"
    except IndexError as e:
        print(e)
        return "Page or Book not found in Database!"
    except Exception as e:
        print(e)
        return "Unknown Error!"


def get_page_of_book(query):
    try:
        book_name, search_query = query.strip().split(":")
        collection = chroma_client.get_collection(name=collection_name,
                                                  embedding_function=embeddings.embed_documents)
        book_collection = chroma_client.get_collection(name=book_collection_name,
                                                       embedding_function=embeddings.embed_documents)

        book_search = book_collection.query(query_texts=book_name, n_results=1,
                                            include=["documents"])
        potential_book = book_search["documents"][0][0]
        where_clause = {
            "$and": [
                {"book": {"$eq": potential_book}},
                {"chunk": {"$eq": int(search_query)}}
            ]
        }
        query_result = collection.query(query_texts=search_query,
                                        n_results=1,
                                        include=["metadatas",
                                                 "documents",
                                                 "distances"],
                                        where=where_clause
                                        )
        return query_result["documents"][0][0]
    except ValueError as ve:
        print(ve)
        return "No data found!"

    except IndexError as e:
        print(e)
        return "Page or Book not found in Database!"
    except Exception as e:
        print(e)
        return "Unknown Error!"

def summarize_book(book_name, llm=None):
    if llm:
        pass
    else:
        return "Summarization not possible. Unset LLM."
