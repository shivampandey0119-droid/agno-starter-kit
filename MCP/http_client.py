import asyncio
from dotenv import load_dotenv

from agno.agent import Agent
from agno.tools.mcp import MCPTools
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat

load_dotenv()

async def run_http_client(message: str) -> None:
    # Connect to the Streamable HTTP server
    # Agno's MCPTools uses 'streamable-http' as the default for HTTP URLs
    mcp_tools = MCPTools(
        url="http://localhost:8002/mcp",
        transport="streamable-http"
    )

    print("Connecting to Math MCP server (Streamable HTTP)...")
    await mcp_tools.connect()
    
    try:
        agent = Agent(
            #model=Gemini(id="gemini-2.5-flash"),
            model=OpenAIChat(id="gpt-4.1-mini"),
            tools=[mcp_tools],
            markdown=True,
            show_tool_calls=True,
        )

        print(f"\nUser: {message}\n")
        await agent.aprint_response(message, stream=True)

    finally:
        await mcp_tools.close()
        print("\nConnection closed.")

if __name__ == "__main__":
    query = "What is 1234 + 5678 and then multiply that result by 2?"
    asyncio.run(run_http_client(query))
