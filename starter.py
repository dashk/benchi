import logging
import sys
import os.path
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.evaluation import DatasetGenerator
from llama_index.llms import Ollama


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

INDEX_STORAGE_PATH = "./storage"
llm = Ollama(model="llama2")
service_context = ServiceContext.from_defaults(llm=llm)
documents = SimpleDirectoryReader("data").load_data()

# check if storage already exists
if not os.path.exists(INDEX_STORAGE_PATH):
    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        show_progress=True)
    # store it for later
    index.storage_context.persist(persist_dir=INDEX_STORAGE_PATH)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_STORAGE_PATH)
    index = load_index_from_storage(storage_context)

data_generator = DatasetGenerator.from_documents(documents)
eval_questions = data_generator.generate_questions_from_nodes()

print(eval_questions)

# either way we can now query the index
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)
