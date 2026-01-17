import asyncio
from dotenv import load_dotenv

from agno.agent import Agent
from agno.tools.mcp import MultiMCPTools
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.models.google import Gemini

load_dotenv()


async def run_agent(message: str) -> None:
    multi_mcp_tools = MultiMCPTools(
        [
            "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt",
            "python mcp/stdio_server.py",   # custom MCP weather server
        ]
    )

    await multi_mcp_tools.connect()

    try:
        agent = Agent(
            model=OpenAIChat(id="gpt-4.1-mini"),
            #model=Groq(id="qwen/qwen3-32b"),
            #model=Gemini(id="gemini-2.5-flash"),
            tools=[multi_mcp_tools],
            markdown=True,
        )

        await agent.aprint_response(message, stream=True)

    finally:
        await multi_mcp_tools.close()


if __name__ == "__main__":
    asyncio.run(
        run_agent(
            "Find Airbnb listings in Delhi for 17 Jan 2026 night and weather in california"
        )
    )
