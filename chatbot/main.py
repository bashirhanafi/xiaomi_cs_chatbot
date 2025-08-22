# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import psycopg
from sqlalchemy import create_engine
from langchain_ollama import ChatOllama
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from tools import CheckOrderStatusTool, CheckSpecification, CheckGuarantee, CheckListProduct
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# --------------------------
# FastAPI setup
# --------------------------
app = FastAPI(title="Xiaomi Customer Service Chatbot API")


# --------------------------
# Database Setup
# --------------------------
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)
conn_info = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
sync_connection = psycopg.connect(conn_info)

table_name = "chat_history"
PostgresChatMessageHistory.create_tables(sync_connection, table_name)


# --------------------------
# Initialize LLM & Embedding Model
# --------------------------
llm = ChatOllama(model="llama3.2:3b", 
                 temperature=0.1)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# --------------------------
# Initialize Chat History & Memory
# --------------------------
session_id = str(uuid.uuid4())
chat_history = PostgresChatMessageHistory(table_name, session_id, sync_connection=sync_connection)

memory = ConversationBufferWindowMemory(memory_key="chat_history", 
                                        k=6, # 3 interaksi sebelumnya 
                                        return_messages=True)

# Load previous messages from DB
for msg in chat_history.messages:
    memory.chat_memory.add_message(msg)

# --------------------------
# Full System Prompt
# --------------------------
system_message = """
Kamu adalah customer service berbasis AI untuk Xiaomi Indonesia. 

Panduan:
- Selalu sapa pengguna dengan kalimat yang hangat, sopan, dan membantu.
- Jawab pertanyaan apapun dengan bahasa natural, tidak hanya yang terkait tools.
- Gunakan tools sesuai kebutuhan dan hasilkan dengan bahasa natural. 
Kamu memiliki beberapa alat (tools):
1. Nama tool: check_order_status, Deskripsi: Untuk menjelaskan status pesanan pengguna., Input: customer_name (str), question (str).
2. Nama tool: check_specification, Deskripsi: untuk menjelaskan spesifikasi dan keunggulan produk., Input: product_name (str), question (str). 
3. Nama tool: check_guarantee, Deskripsi: untuk menjelaskan garansi produk., Input: product_name (str), question (str).
- Jawab pertanyaan dengan respon yang hangat, sopan, dan membantu.
- Jika user ingin bercakap-cakap (halo, kabar, nama, dsb), jawab dengan natural.
- Jangan pernah berhalusinasi.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# tools
check_status = CheckOrderStatusTool()
check_spec = CheckSpecification(embeddings=embeddings)
check_guarantee = CheckGuarantee(embeddings=embeddings)
check_list_product = CheckListProduct()
tools = [check_status, check_spec, check_guarantee, check_list_product]

# agent exectuors
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory)

# --------------------------
# Helper function: save messages
# --------------------------
def save_messages_to_db(user_message: str, ai_message: str):
    chat_history.add_messages([
        HumanMessage(content=user_message),
        AIMessage(content=ai_message)
    ])

# --------------------------
# Pydantic model for request
# --------------------------
class ChatRequest(BaseModel):
    message: str

# --------------------------
# FastAPI endpoint
# --------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    user_input = request.message
    events = agent_executor.stream({"input": user_input})

    ai_response = ""
    for event in events:
        content = event["messages"][-1].content
        if not content:
            continue
        ai_response = str(content)

    save_messages_to_db(user_input, ai_response)
    return {"response": ai_response}

