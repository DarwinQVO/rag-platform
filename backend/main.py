import asyncio
import os
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncpg
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.docstore.document import Document
from supabase import create_client, Client
import tiktoken
from extraction_prompt import EXTRACTION_PROMPT
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB, MAX_PAGES, DEFAULT_MODEL, SIMILARITY_TOP_K

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check required environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    print("WARNING: Missing required environment variables!")
    print("Please set SUPABASE_URL, SUPABASE_KEY, and OPENAI_API_KEY in .env file")
    print("Using mock mode for development...")
    supabase = None
    embeddings = None
    vector_store = None
else:
    # Global connections
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = SupabaseVectorStore(
        supabase, 
        embeddings, 
        table_name="documents",
        query_name="match_documents"
    )

db_pool: asyncpg.Pool = None

async def get_db_pool():
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
    return db_pool

@app.on_event("startup")
async def startup():
    await get_db_pool()

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

@app.get("/")
async def root():
    return {"message": "RAG Platform API", "status": "running", "mock_mode": vector_store is None}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"Archivo muy grande (máx {MAX_FILE_SIZE_MB}MB)")
    
    content = await file.read()
    pdf_reader = PyPDF2.PdfReader(BytesIO(content))
    
    if len(pdf_reader.pages) > MAX_PAGES:
        raise HTTPException(400, f"PDF muy largo (máx {MAX_PAGES} páginas)")
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Split text
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(text)
    
    # Create documents
    doc_id = file.filename.replace(".pdf", "")
    documents = [
        Document(
            page_content=chunk,
            metadata={"doc_id": doc_id, "chunk_id": i, "filename": file.filename}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    # Embed and store
    vector_store.add_documents(documents)
    
    return {"doc_id": doc_id, "chunks": len(chunks), "message": "Documento procesado"}

@app.get("/docs")
async def list_docs():
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT metadata->>'doc_id' as doc_id, metadata->>'filename' as filename FROM documents"
        )
    return [dict(row) for row in rows]

@app.delete("/docs/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its chunks from the vector store and database"""
    if not vector_store:
        raise HTTPException(503, "Vector store not configured")
    
    try:
        # Get document info before deletion
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Check if document exists
            doc_check = await conn.fetchrow(
                "SELECT DISTINCT metadata->>'doc_id' as doc_id, metadata->>'filename' as filename FROM documents WHERE metadata->>'doc_id' = $1 LIMIT 1",
                doc_id
            )
            
            if not doc_check:
                raise HTTPException(404, f"Document '{doc_id}' not found")
            
            # Delete all chunks for this document from database
            deleted_count = await conn.execute(
                "DELETE FROM documents WHERE metadata->>'doc_id' = $1",
                doc_id
            )
            
        # Also delete from vector store (Supabase handles this automatically with the database deletion)
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": doc_check["filename"],
            "message": f"Document '{doc_check['filename']}' deleted successfully",
            "chunks_deleted": int(deleted_count.split()[-1]) if deleted_count else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(500, f"Failed to delete document: {str(e)}")

@app.get("/ask")
async def ask_question(doc_id: str, q: str):
    # Search similar chunks
    results = vector_store.similarity_search(
        q,
        k=SIMILARITY_TOP_K,  # Recuperar más chunks para mejor contexto
        filter={"doc_id": doc_id}
    )
    
    if not results:
        return {"answer": "No encontr� informaci�n relevante", "sources": [], "confidence": 0}
    
    # Build context
    context = "\n\n".join([r.page_content for r in results])
    sources = [{"chunk_id": r.metadata.get("chunk_id", 0), "text": r.page_content[:100] + "..."} for r in results]
    
    # Generate answer
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    prompt = f"""Contexto del documento:
{context}

Pregunta: {q}

Responde en espa�ol, citando las partes relevantes del contexto. S� conciso y preciso."""
    
    response = llm.invoke(prompt)
    
    return {
        "answer": response.content,
        "sources": sources,
        "confidence": 0.85 if len(results) >= 2 else 0.6
    }

@app.get("/extract")
async def extract_structured(doc_id: str):
    # Get all chunks for doc
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content FROM documents WHERE metadata->>'doc_id' = $1 ORDER BY (metadata->>'chunk_id')::int",
            doc_id
        )
    
    if not rows:
        raise HTTPException(404, "Documento no encontrado")
    
    full_text = "\n".join([row["content"] for row in rows])
    tokens = len(tiktoken.encoding_for_model("gpt-4").encode(full_text))
    
    # Choose model based on size
    if tokens > 18000:
        # Would use Gemini here but keeping it simple with truncation
        full_text = full_text[:50000]  # Rough truncation
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = f"""Extrae informaci�n estructurada del siguiente documento:

{full_text}

Devuelve un JSON con:
- quotes: lista de citas importantes con autor y contexto
- entities: personas, organizaciones y lugares mencionados
- metrics: n�meros, estad�sticas y fechas relevantes
- relations: relaciones entre entidades

Formato JSON estricto."""

    async def generate():
        response = llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            data["confidence"] = 0.9
        except:
            data = {
                "quotes": [],
                "entities": [],
                "metrics": [],
                "relations": [],
                "confidence": 0.5,
                "error": "Error al parsear respuesta"
            }
        
        yield json.dumps(data, ensure_ascii=False)
    
    return StreamingResponse(generate(), media_type="application/json")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)