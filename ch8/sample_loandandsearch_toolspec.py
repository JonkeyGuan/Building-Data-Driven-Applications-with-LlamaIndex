from llama_index.core.tools.tool_spec.load_and_search.base import LoadAndSearchToolSpec
from llama_index.tools.database import DatabaseToolSpec
# from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream, ToolCallResult
# from llama_index.llms.openai import OpenAI
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.settings import Settings
import asyncio

async def main():
    Settings.llm = GoogleGenAI(model="gemini-2.0-flash")
    Settings.embed_model = GoogleGenAIEmbedding(
        model_name="models/embedding-001"
    )

    db_tools = DatabaseToolSpec(uri="sqlite:///files/database/employees.db")
    tool_list = db_tools.to_tool_list()
    tools = LoadAndSearchToolSpec.from_defaults(tool_list[0]).to_tool_list()

    agent = ReActAgent(
        tools=tools,
        verbose=True
    )

    ctx = Context(agent)
    handler = agent.run("Who has the highest salary in the Employees table?", ctx=ctx)

    async for ev in handler.stream_events():
        if isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)

    response = await handler
    print("\n\nFinal Answer:", str(response))

if __name__ == "__main__":
    asyncio.run(main())
