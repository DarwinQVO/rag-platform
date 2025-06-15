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
    if not vector_store:
        raise HTTPException(503, "Servicio no disponible. Configure las variables de entorno.")
    
    if file.size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(400, "Archivo muy grande (máx 50MB)")
    
    content = await file.read()
    pdf_reader = PyPDF2.PdfReader(BytesIO(content))
    
    if len(pdf_reader.pages) > 1000:
        raise HTTPException(400, "PDF muy largo (máx 1000 páginas)")
    
    text = ""
    page_map = {}  # Mapear chunks a páginas
    for i, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        start_pos = len(text)
        text += page_text + "\n"
        page_map[start_pos] = i + 1
    
    # Split text with semantic awareness
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=lambda t: len(tiktoken.encoding_for_model("gpt-4").encode(t)),
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    
    # Create documents with enhanced metadata
    doc_id = file.filename.replace(".pdf", "")
    documents = []
    for i, chunk in enumerate(chunks):
        # Find approximate page number for this chunk
        chunk_start = text.find(chunk)
        page_num = 1
        for pos, page in sorted(page_map.items()):
            if pos <= chunk_start:
                page_num = page
            else:
                break
        
        documents.append(Document(
            page_content=chunk,
            metadata={
                "doc_id": doc_id,
                "chunk_id": i,
                "filename": file.filename,
                "page": page_num,
                "chunk_index": f"{i+1}/{len(chunks)}",
                "total_pages": len(pdf_reader.pages)
            }
        ))
    
    # Embed and store
    vector_store.add_documents(documents)
    
    return {
        "doc_id": doc_id,
        "chunks": len(chunks),
        "pages": len(pdf_reader.pages),
        "message": f"Documento procesado: {len(chunks)} fragmentos de {len(pdf_reader.pages)} páginas"
    }

@app.get("/docs")
async def list_docs():
    if not db_pool:
        return []
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT DISTINCT 
                metadata->>'doc_id' as doc_id, 
                metadata->>'filename' as filename,
                MAX((metadata->>'total_pages')::int) as total_pages,
                COUNT(*) as chunk_count
            FROM documents 
            GROUP BY metadata->>'doc_id', metadata->>'filename'"""
        )
    return [dict(row) for row in rows]

@app.get("/ask")
async def ask_question(doc_id: str, q: str):
    if not vector_store:
        raise HTTPException(503, "Servicio no disponible. Configure las variables de entorno.")
    
    # Search similar chunks with enhanced retrieval
    results = vector_store.similarity_search(
        q,
        k=8,  # Más chunks para mejor contexto
        filter={"doc_id": doc_id}
    )
    
    if not results:
        return {"answer": "No encontré información relevante", "sources": [], "confidence": 0}
    
    # Build context with page references
    context_parts = []
    sources = []
    for r in results:
        page = r.metadata.get("page", "?")
        context_parts.append(f"[Página {page}] {r.page_content}")
        sources.append({
            "chunk_id": r.metadata.get("chunk_id", 0),
            "page": page,
            "text": r.page_content[:150] + "...",
            "relevance": getattr(r, 'score', 0.8)
        })
    
    context = "\n\n".join(context_parts)
    
    # Generate answer with structured prompting
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""Contexto del documento (con referencias de página):

{context}

Pregunta: {q}

Instrucciones:
1. Responde en español de forma precisa y completa
2. Cita las páginas relevantes cuando menciones información específica
3. Si encuentras métricas o datos numéricos, resáltalos
4. Si identificas entidades importantes (personas, organizaciones), menciónalas claramente
5. Estructura tu respuesta de forma clara y legible

Respuesta:"""
    
    response = llm.invoke(prompt)
    
    # Calculate confidence based on result quality
    confidence = min(0.95, 0.6 + (len(results) * 0.05))
    if len(results) >= 4:
        confidence = 0.9
    
    return {
        "answer": response.content,
        "sources": sources[:4],  # Top 4 sources
        "confidence": confidence,
        "total_sources": len(results)
    }

@app.get("/extract")
async def extract_structured(doc_id: str):
    if not vector_store:
        raise HTTPException(503, "Servicio no disponible. Configure las variables de entorno.")
    
    # Get all chunks for doc
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content, metadata FROM documents WHERE metadata->>'doc_id' = $1 ORDER BY (metadata->>'chunk_id')::int",
            doc_id
        )
    
    if not rows:
        raise HTTPException(404, "Documento no encontrado")
    
    # Reconstruct with page markers
    full_text = ""
    current_page = 1
    for row in rows:
        page = row["metadata"].get("page", current_page)
        if page != current_page:
            full_text += f"\n\n[PÁGINA {page}]\n\n"
            current_page = page
        full_text += row["content"] + "\n"
    
    tokens = len(tiktoken.encoding_for_model("gpt-4").encode(full_text))
    
    async def generate():
        try:
            # For very long documents, use intelligent chunking
            if tokens > 100000:
                data = await extract_large_document(rows, doc_id)
            elif tokens > 18000 and os.getenv("GEMINI_API_KEY"):
                # Try Gemini for long documents
                data = await extract_with_gemini(full_text)
            else:
                # Use GPT-4 for standard documents
                data = await extract_with_gpt4(full_text, tokens)
            
            # Add document metadata
            data["document_info"] = {
                "doc_id": doc_id,
                "total_tokens": tokens,
                "total_pages": rows[0]["metadata"].get("total_pages", "?"),
                "extraction_method": data.get("model", "unknown")
            }
            
        except Exception as e:
            data = {
                "quotes": [],
                "entities": [],
                "metrics": [],
                "relations": [],
                "confidence": 0.3,
                "error": f"Error durante extracción: {str(e)}",
                "document_info": {"doc_id": doc_id, "total_tokens": tokens}
            }
        
        yield json.dumps(data, ensure_ascii=False, indent=2)
    
    return StreamingResponse(generate(), media_type="application/json")

async def extract_with_gpt4(full_text: str, tokens: int):
    """Extracción con GPT-4"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Si es muy largo, tomar las partes más relevantes
    if tokens > 50000:
        # Tomar inicio, medio y final
        text_parts = []
        text_len = len(full_text)
        text_parts.append(full_text[:15000])  # Inicio
        text_parts.append(full_text[text_len//2-7500:text_len//2+7500])  # Medio
        text_parts.append(full_text[-15000:])  # Final
        full_text = "\n\n[...CONTENIDO OMITIDO...]\n\n".join(text_parts)
    
    prompt = EXTRACTION_PROMPT.format(full_text=full_text)
    response = llm.invoke(prompt)
    
    data = json.loads(response.content)
    data["confidence"] = 0.85 if tokens < 50000 else 0.75
    data["model"] = "gpt-4o"
    
    return validate_extraction(data)

async def extract_with_gemini(full_text: str):
    """Extracción con Gemini para documentos largos"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = EXTRACTION_PROMPT.format(full_text=full_text)
        response = model.generate_content(prompt)
        
        # Limpiar respuesta de Gemini
        response_text = response.text
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        data = json.loads(response_text.strip())
        data["confidence"] = 0.95
        data["model"] = "gemini-1.5-pro"
        
        return validate_extraction(data)
    except Exception as e:
        print(f"Gemini extraction failed: {e}")
        # Fallback to GPT-4
        return await extract_with_gpt4(full_text[:50000], len(full_text))

async def extract_large_document(rows, doc_id):
    """Procesa documentos muy grandes por secciones"""
    # Dividir en secciones manejables
    sections = []
    current_section = []
    current_tokens = 0
    
    for row in rows:
        chunk_tokens = len(tiktoken.encoding_for_model("gpt-4").encode(row["content"]))
        if current_tokens + chunk_tokens > 15000:
            sections.append(current_section)
            current_section = [row]
            current_tokens = chunk_tokens
        else:
            current_section.append(row)
            current_tokens += chunk_tokens
    
    if current_section:
        sections.append(current_section)
    
    # Procesar cada sección
    all_quotes = []
    all_entities = []
    all_metrics = []
    all_relations = []
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    for i, section in enumerate(sections):
        section_text = "\n".join([r["content"] for r in section])
        section_pages = f"{section[0]['metadata'].get('page', '?')}-{section[-1]['metadata'].get('page', '?')}"
        
        prompt = f"""Esta es la sección {i+1} de {len(sections)} del documento (páginas {section_pages}).

{EXTRACTION_PROMPT.format(full_text=section_text)}

Nota: Usa IDs con prefijo s{i+1}_ (ej: s{i+1}_q1, s{i+1}_e1, s{i+1}_m1)"""
        
        response = llm.invoke(prompt)
        section_data = json.loads(response.content)
        
        # Agregar datos de la sección
        all_quotes.extend(section_data.get("quotes", []))
        all_entities.extend(section_data.get("entities", []))
        all_metrics.extend(section_data.get("metrics", []))
        all_relations.extend(section_data.get("relations", []))
    
    # Consolidar y deduplicar entidades
    consolidated_data = {
        "quotes": all_quotes,
        "entities": deduplicate_entities(all_entities),
        "metrics": all_metrics,
        "relations": all_relations,
        "confidence": 0.8,
        "model": "gpt-4o-sectioned",
        "sections_processed": len(sections)
    }
    
    return validate_extraction(consolidated_data)

def deduplicate_entities(entities):
    """Deduplica entidades por nombre similar"""
    unique_entities = {}
    for entity in entities:
        name = entity.get("name", "").lower().strip()
        if name not in unique_entities:
            unique_entities[name] = entity
        else:
            # Merge quote_ids and metric_ids
            unique_entities[name]["quote_ids"] = list(set(
                unique_entities[name].get("quote_ids", []) + 
                entity.get("quote_ids", [])
            ))
            unique_entities[name]["metric_ids"] = list(set(
                unique_entities[name].get("metric_ids", []) + 
                entity.get("metric_ids", [])
            ))
    
    return list(unique_entities.values())

def validate_extraction(data):
    """Valida y enriquece los datos extraídos"""
    # Asegurar que todos los campos existen
    data.setdefault("quotes", [])
    data.setdefault("entities", [])
    data.setdefault("metrics", [])
    data.setdefault("relations", [])
    
    # Crear índices para validación cruzada
    quote_ids = {q.get("id") for q in data["quotes"] if q.get("id")}
    entity_ids = {e.get("id") for e in data["entities"] if e.get("id")}
    metric_ids = {m.get("id") for m in data["metrics"] if m.get("id")}
    
    # Estadísticas de extracción
    data["stats"] = {
        "total_quotes": len(data["quotes"]),
        "total_entities": len(data["entities"]),
        "total_metrics": len(data["metrics"]),
        "total_relations": len(data["relations"]),
        "entities_with_quotes": len([e for e in data["entities"] if e.get("quote_ids")]),
        "quotes_with_metrics": len([q for q in data["quotes"] if q.get("metric_ids")]),
        "cross_referenced_items": len([
            item for collection in [data["quotes"], data["entities"], data["metrics"]]
            for item in collection
            if any([item.get("quote_ids"), item.get("entity_ids"), item.get("metric_ids")])
        ])
    }
    
    # Calcular calidad de extracción
    quality_score = 0
    if data["stats"]["total_quotes"] > 0:
        quality_score += 0.25
    if data["stats"]["total_entities"] > 0:
        quality_score += 0.25
    if data["stats"]["total_metrics"] > 0:
        quality_score += 0.25
    if data["stats"]["cross_referenced_items"] > 0:
        quality_score += 0.25
    
    data["extraction_quality"] = quality_score
    
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)