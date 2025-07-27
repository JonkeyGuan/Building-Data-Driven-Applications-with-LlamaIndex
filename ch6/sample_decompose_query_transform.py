from llama_index.core.indices.query.query_transform.base import DecomposeQueryTransform
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

decompose = DecomposeQueryTransform()
query_bundle = decompose.run(
    "Tell me about buildings in ancient Rome"
)
print(query_bundle.query_str)
