from config import model_cache
from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceBgeEmbeddings

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

all_miniLM_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2", cache_folder=model_cache)
qa_distilbert_embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-distilbert-dot-v1",
                                                         cache_folder=model_cache)
bge_base_embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en",
                                               model_kwargs=model_kwargs,
                                               encode_kwargs=encode_kwargs,
                                               cache_folder=model_cache)
