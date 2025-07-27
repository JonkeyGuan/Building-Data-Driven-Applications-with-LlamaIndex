from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

options = [
    "option 1 this is good for summarization questions",
    "option 2: this is useful for precise definitions",
    "option 3: this is useful for comparing concepts",
]
selector = LLMSingleSelector.from_defaults()

decision = selector.select(
    options, 
    query="What's the definition of space?"
).selections[0]
print(decision.index+1)
print(decision.reason)