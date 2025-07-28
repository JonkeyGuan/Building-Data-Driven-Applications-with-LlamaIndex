from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings
import asyncio

# tool = OnDemandLoaderTool.from_defaults(
#  WikipediaReader(),
#  name="WikipediaReader",
#  description="args: {'pages': [<list of pages>],'query_str': <query>}"           
# )

# agent = OpenAIAgent.from_tools(
#     tools=[tool], 
#     verbose=True
# )
# response = agent.chat(
#     "What were some famous buildings in ancient Rome?")
# print(response)

async def main():
    Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/embedding-001"
    )

    tool = OnDemandLoaderTool.from_defaults(
    WikipediaReader(),
    name="WikipediaReader",
    description="args: {'pages': [<list of pages>],'query_str': <query>}"           
    )

    agent = ReActAgent(
        tools=[tool],
        verbose=True
    )

    ctx = Context(agent)
    handler = agent.run("What were some famous buildings in ancient Rome?", ctx=ctx)

    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)

    response = await handler
    print("\n\nFinal Answer:", str(response))

if __name__ == "__main__":
    asyncio.run(main())