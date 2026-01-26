import requests
from agno.agent import Agent
from agno.media import Audio
from agno.models.google import Gemini

import os
from dotenv import load_dotenv
load_dotenv()

agent = Agent(
    model=Gemini(id="gemini-3-flash-preview"),
    markdown=True,
)

url = "https://agno-public.s3.us-east-1.amazonaws.com/demo_data/QA-01.mp3"

response = requests.get(url)
response.raise_for_status()   # safety check

audio_bytes = response.content

agent.print_response(
    "Give a clear transcript of this audio conversation. "
    "Use Speaker A and Speaker B to identify speakers.",
    audio=[Audio(content=audio_bytes)],
    stream=True,
)
