
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

llm = GoogleGenAI(
    model="gemini-2.0-flash",
)

embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001")

documents = SimpleDirectoryReader('files').load_data()

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("summarize each document in a few sentences")
print(response)


""" 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('files').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("summarize each document in a few sentences")

print(response)
 """