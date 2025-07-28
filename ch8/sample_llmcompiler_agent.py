from llama_index.tools.database import DatabaseToolSpec
from llama_index.packs.agents_llm_compiler import LLMCompilerAgentPack
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

db_tools = DatabaseToolSpec(uri="sqlite:///files//database//employees.db")
agent = LLMCompilerAgentPack(db_tools.to_tool_list(), llm=llm)

response = agent.run(
    "List the HR department employee "
    "with the highest salary "
)

# am schimbat doar importul . nu mai e necesar download-ul inainte
# de mentionat: pip install llama-index-packs-agents-llm-compiler