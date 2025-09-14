import chainlit as cl
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools

import os
from dotenv import load_dotenv
load_dotenv()



agent=Agent(
    model=Groq(id="qwen/qwen3-32b"),
    description="You are an assistant please reply based ont he question,use tables ,block diagram to display data and provide some best sources",
    tools=[DuckDuckGoTools()],
    instructions="Search provided tools and provide correct responses",
    show_tool_calls=True,
    markdown=True
)



@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I'm your teaching  assistant. Ask me any question."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        response = agent.run(message.content)
        await cl.Message(content=response.content).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()
    
