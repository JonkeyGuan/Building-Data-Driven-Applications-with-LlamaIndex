from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticMultiSelector, LLMSingleSelector
from llama_index.core import SummaryIndex, SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

documents = SimpleDirectoryReader("files").load_data()

title_extractor = TitleExtractor()
for doc in documents:
    title_metadata = title_extractor.extract([doc])
    doc.metadata.update(title_metadata[0])

indexes = []
query_engines = []
tools = []

for doc in documents:
    document_title = doc.metadata['document_title']
    index = SummaryIndex.from_documents([doc])
    query_engine = index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        description=f"Contains data about {document_title}",
    )
    indexes.append(index)
    query_engines.append(query_engine)
    tools.append(tool)

qe = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    # selector=PydanticMultiSelector.from_defaults(),
    query_engine_tools=tools
)

response = qe.query(
    "Tell me about Rome and dogs"
)
print(response)
