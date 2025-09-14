import chainlit as cl
from agno.agent import Agent
from agno.knowledge.url import UrlKnowledge
from agno.models.google import Gemini
from agno.storage.sqlite import SqliteStorage
from agno.vectordb.lancedb import LanceDb, SearchType
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Custom embedder
class LocalHFEmbedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str):
        return self.model.encode(text).tolist()

    def get_embedding_and_usage(self, text: str):
        emb = self.model.encode(text).tolist()
        return emb, {"tokens": len(text.split())}

# Load knowledge
knowledge = UrlKnowledge(
    urls=["https://tameson.com/pages/actuator"],
    vector_db=LanceDb(
        uri="tmp/lancedb",
        table_name="ac_docs",
        search_type=SearchType.vector,
        embedder=LocalHFEmbedder(),
    ),
)

# Storage
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")

# Agent
agent = Agent(
    name="Agno Assist",
    model=Gemini(id="gemini-2.5-flash"),
    instructions=[
        "Search your knowledge before answering the question.",
        "Only include the output in your response. No other text.",
    ],
    knowledge=knowledge,
    storage=storage,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
    markdown=True,
)

# Load knowledge base at startup
agent.knowledge.load(recreate=False)

# Chainlit handlers
@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I'm your Agno-based assistant. Ask me anything about actuators or related topics."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        # Run the agent and capture the response
        response = agent.run(message.content)
        # Send the response to Chainlit UI
        await cl.Message(content=response.content).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()


