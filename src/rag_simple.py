from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import pandas as pd
import xlsxwriter as xw

# import any embedding model from HuggingFace
Settings.embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Setting up the Knowledge Base (which will be the foundation for the RAG data source)
Settings.llm = None
Settings.chunk_size = 256
Settings.chunk_overlap = 25

documents = SimpleDirectoryReader("articles").load_data()
print(f"the number of entries in the documents is {len(documents)}")
i = 0
for doc in documents:
    print(f"the datatype of doc is {type(doc)}")
    # print(f"this is the content of doc.text: {doc.text}")
    if (
        "Member-only story" in doc.text
        or "The Data Entrepreneur" in doc.text
        or "min read" in doc.text
    ):
        print(f"removing doc_{i}")
        i += 1
        documents.remove(doc)
print(f"the number of entries in the documents after exclusions is {len(documents)}")
pd_documents = pd.DataFrame(documents)
pd_documents.to_excel("articles.xlsx", engine="xlsxwriter")
# This is the knowledge data source index
index = VectorStoreIndex.from_documents(documents)
# set number of docs to retrieve.
top_k = 3
# configurer the retriever
retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
# assemble the query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
)
query = "what is fat-tailedness"
response = query_engine.query(query)
# reformat response:
context = "Context\n"
for k in range(top_k):
    context += f"Document {k+1}:\n{response.source_nodes[k].text}\n"
# Since there are characters in the response that fail in the print,
# we need to encode the response to utf-8
print(context.encode("utf-8"))
