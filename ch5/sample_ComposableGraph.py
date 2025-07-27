from llama_index.core import ComposableGraph, SimpleDirectoryReader, TreeIndex, SummaryIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

documents = SimpleDirectoryReader("files").load_data()
index1 = TreeIndex.from_documents([documents[0]])
index2 = TreeIndex.from_documents([documents[1]])
summary1 = "A short introduction to ancient Rome"
summary2 = "Some facts about dogs"

graph = ComposableGraph.from_indices(
    SummaryIndex, [index1, index2], 
    index_summaries=[summary1, summary2]
)
query_engine = graph.as_query_engine()

response = query_engine.query("What can you tell me?")
print(response)
