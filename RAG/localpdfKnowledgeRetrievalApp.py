import chainlit as cl
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.storage.sqlite import SqliteStorage
from sentence_transformers import SentenceTransformer
from agno.models.google import Gemini
from agno.models.openai import OpenAIChat
import os
from dotenv import load_dotenv

load_dotenv()

# --- Local HF embedder wrapper ---
class LocalHFEmbedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.dimensions = self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str):
        return self.model.encode(text).tolist()

    def get_embedding_and_usage(self, text: str):
        emb = self.model.encode(text).tolist()
        return emb, {"tokens": len(text.split())}

# --- Build a local PDF knowledge base ---
pdf_knowledge_base = PDFKnowledgeBase(
    #path=r"C:\Users\lenovo\Downloads\dcn.pdf",
    path=r"C:\Users\lenovo\OneDrive\Desktop\AGNOAGENTS\RAG\dcn.pdf",# 👈 your PDF
    vector_db=LanceDb(                        # 👈 local LanceDB
        uri="tmp/lancedb",
        table_name="local_pdf_document",
        search_type=SearchType.vector,
        embedder=LocalHFEmbedder(),           # 👈 local embeddings
    ),
    reader=PDFReader(chunk=True),
)

# Load / index PDF once
pdf_knowledge_base.load(recreate=False)

# Storage for chat sessions
storage = SqliteStorage(table_name="dcn_session", db_file="tmp/agent.db")

# Use any model you like for answering (Gemini here still calls API)
# If you want zero API calls, you’d need a local text generation model instead.

agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    #model=OpenAIChat(id="gpt-4.1-mini"), # 👈 This is still online model
    storage=storage,
    knowledge=pdf_knowledge_base,
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_runs=3,
    add_references=True,
    search_knowledge=True,
    markdown=True,
)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="👋 Hi! I'm your Geeta assistant. Ask me about any chapter or verse."
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    try:
        from asyncio import to_thread
        response = await to_thread(agent.run, message.content)
        await cl.Message(content=response.content).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()
