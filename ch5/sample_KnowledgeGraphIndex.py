from llama_index.core import KnowledgeGraphIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

documents = SimpleDirectoryReader("files").load_data()
index = KnowledgeGraphIndex.from_documents(documents, max_triplets_per_chunk=2, use_async=True)
query_engine = index.as_query_engine()
response = query_engine.query("Tell me about dogs.")
print(response)
