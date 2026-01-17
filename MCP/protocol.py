import asyncio
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.tools.mcp import MultiMCPTools
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
load_dotenv()


async def run_agent(message: str) -> None:
    """Run the Airbnb and Google Maps agent with the given message."""

    

    multi_mcp_tools = MultiMCPTools(
        [
            "npx -y @openbnb/mcp-server-airbnb --ignore-robots-txt",
        ],
        
    )
    await multi_mcp_tools.connect()
    agent = Agent(
        model=OpenAIChat(id="gpt-4.1-mini"),
        tools=[multi_mcp_tools],
        markdown=True,
    )
    await agent.aprint_response(message, stream=True)
    await multi_mcp_tools.close()
# Example usage
if __name__ == "__main__":
    
    asyncio.run(
        run_agent(
            "What listings are available in California for  17 jan 2026 night and weather in delhi ?"
        )
    )