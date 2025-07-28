from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

documents = SimpleDirectoryReader("files").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine=index.as_query_engine()
chat_history = [
    ChatMessage(
        role="user",
        content="Arch of Constantine is a famous"
        "building in Rome"
    ),
    ChatMessage(
        role="user",
        content="The Pantheon should not be "
        "regarded as a famous building"
    ),
]

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    chat_history=chat_history
)
response = chat_engine.chat(
    "What are two of the most famous structures in ancient Rome?"
)
print(response)
