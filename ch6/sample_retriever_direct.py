from llama_index.core import SummaryIndex, SimpleDirectoryReader
from llama_index.core.retrievers import SummaryIndexEmbeddingRetriever
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

documents = SimpleDirectoryReader("files").load_data()
summary_index = SummaryIndex.from_documents(documents)
retriever = SummaryIndexEmbeddingRetriever(
    index=summary_index
)
result = retriever.retrieve("Tell me about ancient Rome")
print(result[0].text)
