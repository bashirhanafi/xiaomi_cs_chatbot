# import package
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import Type, Any, Optional
from langchain.tools import BaseTool
import ast, os
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from sqlalchemy import text, create_engine
import os, ast, psycopg
from dotenv import load_dotenv

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

# --------------------------
# Tool 1: Check List of Product
# --------------------------
class OrderListInput(BaseModel):
    product_name: str = Field(..., description="Nama produk untuk cek mengecek ketersediaan produk")
    variant: Optional[str] = Field(..., description='Variant dari produk untuk mengecek ketersediaan produk')
    question: str = Field(..., description="query dari user")

class CheckListProduct(BaseTool):
    name: str = "check_list_product"
    description: str = "Hanya gunakan tool ini jika pengguna ingin mengecek ketersediaan product_name dalam database."
    args_schema: Type[BaseModel] = OrderListInput

    def _run(self, product_name: str, question: str, variant: str = None):
        if not any(k in question.lower() for k in [
            'ketersediaan', 'tersedia', 'ada?', 'apakah ada?', 
            'stok', 'ready', 'available', 'order', 'dapat diorder'
        ]):
            return None

        with engine.connect() as conn:
            if variant:  
                result = conn.execute(
                    text("""
                        SELECT product_name, variant, qty, price 
                        FROM products 
                        WHERE product_name = :product_name
                          AND variant = :variant
                    """),
                    {"product_name": product_name, "variant": variant}
                )
            else:  
                result = conn.execute(
                    text("""
                        SELECT product_name, variant, qty, price 
                        FROM products 
                        WHERE product_name = :product_name
                    """),
                    {"product_name": product_name}
                )

        return [dict(row._mapping) for row in result]

# --------------------------
# Tool 2: Check Order Status
# --------------------------
class OrderStatusInput(BaseModel):
    customer_name: str = Field(..., description="Nama customer untuk cek status pesanan")
    question: str = Field(..., description="query dari user")

class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = "Hanya gunakan tool ini jika pengguna menanyakan status pesanan tertentu berdasarkan customer_name."
    args_schema: Type[BaseModel] = OrderStatusInput 

    def _run(self, customer_name: str, question: str):
        if not any(k in question.lower() for k in ["status", "pesanan", "status pesanan", "atas nama"]):
            return None
        with engine.connect() as conn:
            result = conn.execute(
                text("""SELECT 
                    c.customer_name,
                    o.id AS order_id,
                    o.order_date,
                    p.product_name,
                    p.variant,
                    o.qty,
                    o.total,
                    o.status
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                JOIN products p ON o.product_id = p.id
                WHERE c.customer_name = :customer_name;
            """),
            {"customer_name": customer_name}
        ).fetchone()

        if result:
            return f"Pesanan oleh {result.customer_name} memiliki status {result.status}"
        return f"Tidak ada pesanan atas nama {customer_name}"

    async def _arun(self, customer_name: str):
        raise NotImplementedError("Async not supported")
    
# --------------------------
# Tool 3: Check Specification
# --------------------------
class CheckSpecificationInput(BaseModel):
    product_name: str = Field(..., description="Nama produk untuk cek spesifikasi produk")
    question: str = Field(..., description="query dari user")

class CheckSpecification(BaseTool):
    name: str = "check_specification"
    description: str = "Hanya gunakan tool ini jika pengguna menanyakan tentang spesifikasi atau keunggulan produk."
    args_schema: Type[BaseModel] = CheckSpecificationInput 

    embeddings: Any = Field(default=None)

    def _run(self, product_name: str, question: str):
        if not any(k in question.lower() for k in ["spesifikasi", "keunggulan", "feature", "fitur", "produk", "spek", "kelebihan", "advagantes", "kekurangan"]):
            return None
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT specification FROM products WHERE product_name = :product_name LIMIT 1"),
                    {"product_name": product_name}
            ).fetchone()
        spec = result[0]

        # embeddings
        store = InMemoryVectorStore(self.embeddings)

        # loader
        loader = PyPDFLoader(os.path.join("documents", spec))
        docs = loader.load()

        # chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # index chunk
        _ = store.add_documents(documents=all_splits)

        # Define state for application
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        # Define application steps
        def retrieve(state: State):
            retrieved_docs = store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        result = graph.invoke({"question": self.description + "(" + product_name + ")"})
        return result["context"]
    
    async def _arun(self, product_name: str):
        raise NotImplementedError("Async not supported")

# --------------------------
# Tool 4: Check Guarantee
# --------------------------
class CheckGuaranteeInput(BaseModel):
    product_name: str = Field(..., description="Nama produk untuk cek garansi produk")
    question: str = Field(..., description="query dari user")

class CheckGuarantee(BaseTool):
    name: str = "check_guarantee"
    description: str = "Hanya gunakan tool ini jika pengguna menanyakan tentang garansi produk"
    args_schema: Type[BaseModel] = CheckGuaranteeInput 

    embeddings: Any = Field(default=None)

    def _run(self, product_name: str, question: str):
        if not any(k in question.lower() for k in ["garansi", "guarantee", "cek garansi", "pengembalian produk", "proses pengembalian"]):
            return None

        # embeddings
        store = InMemoryVectorStore(self.embeddings)

        # loader
        loader = PyPDFLoader("documents\garansi.pdf")
        docs = loader.load()

        # chunking
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)

        # index chunk
        _ = store.add_documents(documents=all_splits)

        # Define state for application
        class State(TypedDict):
            question: str
            context: List[Document]
            answer: str

        # Define application steps
        def retrieve(state: State):
            retrieved_docs = store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        result = graph.invoke({"question": question})
        return result["context"]
    
    async def _arun(self, product_name: str):
        raise NotImplementedError("Async not supported")
    
