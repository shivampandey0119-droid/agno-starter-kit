from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.dalle import DalleTools
from agno.tools.duckduckgo import DuckDuckGoTools  # Optional, for research if needed
import os
from dotenv import load_dotenv
load_dotenv()

agent = Agent(
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[DalleTools()],  # Enables DALL·E image generation
    markdown=True,
)

agent.print_response(
    "Generate an image of a futuristic cityscape at sunset with flying cars.",
    
)
