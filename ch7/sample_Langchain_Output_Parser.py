from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from pydantic import BaseModel
from typing import List
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

nodes = [
    TextNode(
        text="Roses have vibrant colors and smell nice."),
    TextNode(
        text="Oak trees are tall and have green leaves."),
]

schemas = [
    ResponseSchema(
        name="answer", 
        description=(
            "answer to the user's question"
        )
    ),
    ResponseSchema(
        name="source",
        description=(
            "the source text used to answer the user's question, "
            "should be a quote from the original prompt."
        )
    )
]

lc_parser = StructuredOutputParser.from_response_schemas(schemas)
output_parser = LangchainOutputParser(lc_parser)

# llm = OpenAI(output_parser=output_parser)
llm = GoogleGenAI(model="gemini-2.0-flash", output_parser=output_parser)

index = VectorStoreIndex(nodes=nodes)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query(
    "Are oak trees small? yes or no",
)
print(response)
