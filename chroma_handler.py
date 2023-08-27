import os
import re

import numpy as np
import chromadb
from chromadb.config import Settings
from config import docs_cache, collection_name, context_len, overlap, \
    book_collection_name, clusters, memory_cache, conversation_cache, \
    summarized_history_collection, conversation_collection, memory_collection
from embeddings import bge_base_embeddings as embeddings
import pypdfium2 as pdfium
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from sklearn.cluster import KMeans

document_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=docs_cache
))
memory_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=memory_cache
))
conversation_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=conversation_cache
))


def process_pdf(pdf_path):
    metadata = pdf_path.split(os.sep)[-1][:-4]
    doc = pdfium.PdfDocument(pdf_path)
    full_text = ""
    for page_num in range(len(doc)):
        full_text += doc.get_page(page_num).get_textpage().get_text_range()
    doc.close()
    collection = document_client.get_or_create_collection(name=collection_name,
                                                          embedding_function=embeddings.embed_documents,
                                                          metadata={"hnsw:space": "ip"})
    book_collection = document_client.get_or_create_collection(name=book_collection_name,
                                                               embedding_function=embeddings.embed_documents,
                                                               metadata={"hnsw:space": "ip"})
    id_offset = collection.count()
    doc_collection = []
    metadata_collection = []
    ids_collection = []
    i = idx = 0
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
                        metadatas=[{'Total_pages': len(doc_collection)}],
                        ids=[str(book_collection.count())])
    document_client.persist()
    return


def put_conversation(conversation, summary):
    conversations = conversation_client.get_or_create_collection(name=conversation_collection,
                                                                 embedding_function=embeddings.embed_documents,
                                                                 metadata={"hnsw:space": "ip"})
    summaries = conversation_client.get_or_create_collection(name=summarized_history_collection,
                                                             embedding_function=embeddings.embed_documents,
                                                             metadata={"hnsw:space": "ip"})
    conversations.add(documents=[conversation],
                      metadatas=[{'id': conversations.count()}],
                      ids=[str(conversations.count())])
    summaries.add(documents=[summary],
                  metadatas=[{'id': summaries.count()}],
                  ids=[str(summaries.count())])
    conversation_client.persist()
    return


def get_last_chat_summary():
    try:
        summaries = conversation_client.get_collection(name=summarized_history_collection,
                                                       embedding_function=embeddings.embed_documents)
        last_summary = summaries.query(query_texts="",
                                       n_results=1,
                                       include=["documents"],
                                       where={"id": summaries.count() - 1})
        return last_summary['documents'][0][0]
    except:
        return ""


def put_memory(memories):
    memory_db = memory_client.get_or_create_collection(name=memory_collection,
                                                       embedding_function=embeddings.embed_documents,
                                                       metadata={"hnsw:space": "ip"})
    current_memory_count = memory_db.count()
    metadata = []
    ids = []
    i = 0
    shortlisted_memory = []
    for memory in memories:
        tool_usage = re.findall("`(.*?)`", memory)
        if tool_usage[0] == 'Introspection':
            continue
        # closest_memory = memory_db.query(query_texts=memory, n_results=1, include=['distances'])
        # print(closest_memory['distances'][0][0])
        metadata.append({'id': current_memory_count + i})
        ids.append(str(current_memory_count + i))
        shortlisted_memory.append(memory)
        i += 1
    if shortlisted_memory:
        memory_db.add(documents=shortlisted_memory,
                      metadatas=metadata,
                      ids=ids)
        memory_client.persist()
    return


def process_query(query):
    num_records = 3
    try:
        collection = document_client.get_collection(name=collection_name,
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
        collection = document_client.get_collection(name=collection_name,
                                                    embedding_function=embeddings.embed_documents)
        book_collection = document_client.get_collection(name=book_collection_name,
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
        collection = document_client.get_collection(name=collection_name,
                                                    embedding_function=embeddings.embed_documents)
        book_collection = document_client.get_collection(name=book_collection_name,
                                                         embedding_function=embeddings.embed_documents)

        book_search = book_collection.query(query_texts=book_name, n_results=1,
                                            include=["documents", "metadatas", "distances"])
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


def search_memories(query):
    try:
        collection = memory_client.get_collection(name=memory_collection,
                                                  embedding_function=embeddings.embed_documents)

        memory_matches = collection.query(query_texts=query, n_results=3,
                                          include=["documents"])
        potential_memories = '\n\n'.join(memory_matches["documents"][0])

        return potential_memories
    except ValueError as ve:
        print(ve)
        return "No data found!"


def book_finder(query):
    try:
        collection = document_client.get_collection(name=collection_name,
                                                    embedding_function=embeddings.embed_documents)

        query_result = collection.query(query_texts=query,
                                        n_results=1,
                                        include=["metadatas",
                                                 "documents",
                                                 "distances"]
                                        )
        return query_result["metadatas"][0][0]["book"]
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
        book_collection = document_client.get_collection(name=book_collection_name,
                                                         embedding_function=embeddings.embed_documents)

        book_search = book_collection.query(query_texts=book_name, n_results=1,
                                            include=["documents", "metadatas", "distances"])
        potential_book = book_search["documents"][0][0]
        total_pages = book_search["metadatas"][0][0]['Total_pages']
        collection = document_client.get_collection(name=collection_name,
                                                    embedding_function=embeddings.embed_documents)
        all_pages = collection.query(query_texts="", n_results=total_pages,
                                     include=["documents", "embeddings", "metadatas"],
                                     where={"book": potential_book})
        all_docs = [page for page in all_pages['documents'][0]]
        all_meta = [meta['chunk'] for meta in all_pages['metadatas'][0]]
        all_embeddings = np.asarray(all_pages['embeddings'][0])
        kmeans = KMeans(n_clusters=clusters, random_state=451).fit(all_embeddings)
        # page_wise_classes = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        most_relevant_pages = []
        relevant_page_orders = []
        for center in cluster_centers:
            similarities = all_embeddings @ center
            cluster_best_match = np.argmax(similarities)
            most_relevant_pages.append(all_docs[cluster_best_match])
            relevant_page_orders.append(all_meta[cluster_best_match])
        sorted_documents = sorted(most_relevant_pages, key=lambda x: relevant_page_orders[most_relevant_pages.index(x)])
        sorted_documents = [Document(page_content=doc) for doc in sorted_documents]
        map_prompt = """
        You will be given a single page of a book. This section will be enclosed in triple backticks (```)
        Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
        Your response should be as brief as possible and fully encompass what was said in the page. Names, dates, facts
        should not be lost while summarizing.

        ```{text}```
        FULL SUMMARY:
        """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
        map_chain = load_summarize_chain(llm=llm,
                                         chain_type="stuff",
                                         prompt=map_prompt_template,
                                         verbose=True)
        summaries = []
        for sdoc in sorted_documents:
            summary = map_chain.run([sdoc])
            summaries.append(summary.strip())
        summaries = "\n".join(summaries)
        combine_prompt = """
        You will be given a series of summaries from a book. The summaries will be enclosed in triple backticks (```) \
        Your goal is to give a brief verbose summary of what happened in the story.  \
        The summary should also include names, dates, facts, etc. if they are available. \
        The reader should be able to fully grasp what happened in the book.

        ```{text}```
        VERBOSE SUMMARY:
        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
        reduce_chain = load_summarize_chain(llm=llm,
                                            chain_type="stuff",
                                            prompt=combine_prompt_template,
                                            verbose=True  # Set this to true if you want to see the inner workings
                                            )
        output = reduce_chain.run([Document(page_content=summaries)])
        return output
    else:
        return "Summarization not possible. Unset LLM."
