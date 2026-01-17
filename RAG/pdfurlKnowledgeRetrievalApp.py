import chainlit as cl
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from agno.agent import Agent
from agno.models.google import Gemini  # still uses Gemini for answering; can replace later
from agno.storage.sqlite import SqliteStorage
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb, SearchType  # ✅ Use LanceDb instead of PgVector

load_dotenv()

# --- Local HuggingFace embedder wrapper ---
class LocalHFEmbedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str):
        return self.model.encode(text).tolist()

    def get_embedding_and_usage(self, text: str):
        emb = self.model.encode(text).tolist()
        return emb, {"tokens": len(text.split())}

# Create knowledge base with LanceDb (no db_url needed)
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=LanceDb(
        uri="tmp/lancedb",  # local folder for LanceDB
        table_name="thai_recipe",
        search_type=SearchType.vector,
        embedder=LocalHFEmbedder(),  # ✅ Local embedder here
    )
)

# SQLite for chat session storage
storage = SqliteStorage(table_name="agent_sessions", db_file="tmp/agent.db")

# Load knowledge base (will replace duplicates instead of inserting new ones)
knowledge_base.load(upsert=True)

# Create the RAG agent
agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),  # uses Gemini for answering; can swap to local model if desired
    storage=storage,
    knowledge=knowledge_base,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
    add_references=True,
    search_knowledge=False,
    markdown=True,
)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I'm your thai recipes assistant."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        # Since agent.run is sync, run it in a separate thread for Chainlit async
        from asyncio import to_thread
        response = await to_thread(agent.run, message.content)
        await cl.Message(content=response.content).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()
