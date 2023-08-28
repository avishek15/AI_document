import chromadb
from chromadb.config import Settings
from src.config import docs_cache, collection_name, context_len, overlap, \
    book_collection_name, clusters, memory_cache, conversation_cache, \
    summarized_history_collection, conversation_collection, memory_collection
from src.embeddings import bge_base_embeddings as embeddings

document_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./remote_dbs/docs_cache"
))
# memory_client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory=memory_cache
# ))
conversation_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./remote_dbs/conversation_cache"
))

book_collection = document_client.get_collection(name=book_collection_name,
                                                 embedding_function=embeddings.embed_documents)

conversation_db = conversation_client.get_collection(name=conversation_collection,
                                                     embedding_function=embeddings.embed_documents)

print(book_collection.count())
