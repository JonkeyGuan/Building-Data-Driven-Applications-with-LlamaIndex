from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

reader = SimpleDirectoryReader('files/other')
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(retriever_mode='default')
nodes = retriever.retrieve(
    "What did Fluffy found in the gentle stream?"
)

print('Initial nodes:')
for node in nodes:
    print(f"Node: {node.node_id} - Score: {node.score}")

pp = SimilarityPostprocessor(
    nodes=nodes, 
    similarity_cutoff=0.86
)

remaining_nodes = pp.postprocess_nodes(nodes)
print('Remaining nodes:')
for node in remaining_nodes:
    print(f"Node: {node.node_id} - Score: {node.score}")
