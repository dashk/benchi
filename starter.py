from dataclasses import dataclass
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
from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator
from llama_index.llms import Ollama


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

INDEX_STORAGE_PATH = "./storage"
LIMIT = 1

llm = Ollama(model="mistral:7b")
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

data_generator = DatasetGenerator.from_documents(documents, service_context=service_context)
eval_questions = data_generator.generate_questions_from_nodes(num=1)

print(eval_questions)

evaluator = RelevancyEvaluator(service_context=service_context)

# create vector index
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
query_engine = vector_index.as_query_engine()

questions_and_results = []

@dataclass
class Question:
    question: str
    answer: str
    reasoning: str
    score: float


for i, question in enumerate(eval_questions):
    if i >= LIMIT:
        break
    logging.info("Processing %i/%i", i, len(eval_questions))

    logging.info("Query: %s", question)
    response_vector = query_engine.query(question)
    logging.info("Vector response: %s", response_vector)

    logging.info("Evaluation")
    eval_result = evaluator.evaluate_response(
        query=question, response=response_vector
    )
    logging.info("Result: %s", eval_result)
    questions_and_results.append(
        Question(
            question=eval_result.query,
            answer=eval_result.response,
            reasoning=eval_result.feedback,
            score=eval_result.score,
        )
    )

# Output evaluation results to a JSONL file
with open("evaluation_results.jsonl", "w") as f:
    for q in questions_and_results:
        f.write(f"{q}\n")

# either way we can now query the index
# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)
