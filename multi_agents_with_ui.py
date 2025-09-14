import chainlit as cl
from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv

#save your API KEYS in .env file
# Load environment variables
load_dotenv()
#otherwise set API KEYS directly
#GOOGLE_API_KEY="your google api key"
#GROQ_API_KEY="your groq api key"


# Initialize agents
web_agent=Agent(
    name="Web Agent",
    role="search the web for information",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[DuckDuckGoTools()],
    instructions="Always include the sources",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Groq(id="qwen/qwen3-32b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,stock_fundamentals=True,company_info=True)],
    instructions="Use tables to display data",
    show_tool_calls=True,
    markdown=True,
)

agent_team=Agent(
    team=[web_agent,finance_agent],
    model=Groq(id="qwen/qwen3-32b"),
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I'm your stock market assistant. Ask me about companies, market trends, or investment advice."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        response = agent_team.run(message.content)
        await cl.Message(content=response.content).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()