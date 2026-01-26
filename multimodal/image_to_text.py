import requests
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image
import os
from dotenv import load_dotenv

load_dotenv()


agent = Agent(
    model=Gemini(id="gemini-3-flash-preview"),
    #tools=[DuckDuckGoTools()],--not supported
    markdown=True,
)

image_url = "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}

response = requests.get(image_url, headers=headers)
response.raise_for_status()

image_bytes = response.content

agent.print_response(
    "Describe this image ",
    images=[Image(content=image_bytes)],
    stream=True,
)
