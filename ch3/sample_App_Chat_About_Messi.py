from llama_index.core import Document, SummaryIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.llms.google_genai import GoogleGenAI
import logging

logging.basicConfig(level=logging.DEBUG)

llm = GoogleGenAI(
    model="gemini-2.0-flash",
)

loader = WikipediaReader()
documents = loader.load_data(pages=["Messi Lionel"])
parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)
index = SummaryIndex(nodes)
query_engine = index.as_query_engine(llm=llm)
print("Ask me anything about Lionel Messi!")

while True:
    question = input("Your question: ")
    if question.lower() == "exit":
        break
    response = query_engine.query(question)
    print(response)
