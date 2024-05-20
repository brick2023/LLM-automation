from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import StorageContext, load_index_from_storage

from rich import print
import os
import time

def text_embedding():
    # 設定
    Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/paraphrase-xlm-r-multilingual-v1") # 支援多國語言
    # Settings.embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large") # alternative model

    Settings.llm = None
    Settings.chunk_size = 256
    Settings.chunk_overlap = 25

    # articles available here: {add GitHub repo}
    print("Loading articles...")
    # 將 src 底下的所有資料夾地下的 plain_text 資料夾底下的所有檔案讀取進來
    # 只讀取第一層資料夾
    dir_list = [dirnames for dirpath, dirnames, files in os.walk("/home/brick2/platform2024/src/")][0]
    try:
        dir_list.remove("test_index")
    except:
        print("尚未建立索引")
    documents_path = ["/home/brick2/platform2024/src/" + dirname + "/plain_text/" for dirname in dir_list]
    print(f"documents_path: {documents_path}")
    print(f"dir_list: {dir_list}")

    documents = []
    for path in documents_path:
        reader = SimpleDirectoryReader(path).load_data()
        documents.extend(reader)
    print(len(documents), "articles loaded.")

    # create index
    # 檢查索引是否存在，如果不存在，則建立索引，否則載入索引並附加新文件
    index_path = "/home/brick2/platform2024/src/test_index"
    if not os.path.exists(index_path):
        print("Creating index...")
        index = VectorStoreIndex.from_documents(documents) # 用來建立索引
        index.storage_context.persist(index_path)
    else:
        print("Loading index...")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)

    print(f"index {index}")