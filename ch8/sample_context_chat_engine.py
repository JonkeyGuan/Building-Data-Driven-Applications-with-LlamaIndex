from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

docs = SimpleDirectoryReader(input_dir="files").load_data()
index = VectorStoreIndex.from_documents(docs)
chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=(
        "You’re a chatbot, able to talk about "
        "general topics, as well as answering specific "
        "questions about ancient Rome."
    ),
)
chat_engine.chat_repl()
