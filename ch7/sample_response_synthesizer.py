from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core import get_response_synthesizer
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

nodes = [ 
    TextNode(text=
        "The town square clock was built in 1895"
    ), 
    TextNode(text=
        "A turquoise parrot lives in the Amazon"
    ), 
    TextNode(text=
        "A rare orchid blooms only at midnight"
    ), 
] 

node_with_score_list = [NodeWithScore(node=node) for node in nodes] 
synth = get_response_synthesizer( 
    response_mode="refine", 
    use_async=False, 
    streaming=False, 
) 

response = synth.synthesize( 
    "When was the clock built?", 
    nodes=node_with_score_list 
) 
print(response) 