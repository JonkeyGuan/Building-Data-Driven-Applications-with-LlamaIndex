import asyncio
from llama_index.core import KeywordTableIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings

Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/embedding-001"
)

async def retrieve(retriever, query, label):
    response = await retriever.aretrieve(query)
    print(f"{label} retrieved {str(len(response))} nodes")

async def main():
    reader = SimpleDirectoryReader('files')
    documents = reader.load_data()
    index = KeywordTableIndex.from_documents(documents)
    retriever1 = index.as_retriever(
retriever_mode='default'
)
    retriever2 = index.as_retriever(
retriever_mode='simple'
)
    query = "Where is the Colosseum?"

    await asyncio.gather(
        retrieve(retriever1, query, '<llm>'),
        retrieve(retriever2, query, '<simple>')
    )
asyncio.run(main())
