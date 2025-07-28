from llama_index.core.postprocessor import KeywordNodePostprocessor
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

nodes = [
    TextNode(
        text="Entry no: 1, <SECRET> - Attack at Dawn"
    ),
    TextNode(
        text="Entry no: 2, <RESTRICTED> - Go to point Bravo"
    ),
    TextNode(
        text="Entry no: 3, <PUBLIC> - Roses are Red"
    ),
]

node_with_score_list = [
    NodeWithScore(node=node) for node in nodes
]
pp = KeywordNodePostprocessor(
    exclude_keywords=["SECRET", "RESTRICTED"]
)
remaining_nodes = pp.postprocess_nodes(
    node_with_score_list
)
print('Remaining nodes:')
for node_with_score in remaining_nodes:
    node = node_with_score.node
    print(f"Text: {node.text}")
