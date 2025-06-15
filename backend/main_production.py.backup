import asyncio
import os
import re
from typing import List, Dict, Any, Optional
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
from extraction_prompt_v2 import EXTRACTION_PROMPT_V2

load_dotenv()

app = FastAPI(title="RAG Platform API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment validation
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY]):
    print("❌ CRITICAL: Missing required environment variables!")
    print("Required: SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY")
    print("Optional: GEMINI_API_KEY (for large documents)")
    supabase = None
    embeddings = None
    vector_store = None
else:
    print("✅ Environment variables configured")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = SupabaseVectorStore(
        supabase, 
        embeddings, 
        table_name="documents",
        query_name="match_documents"
    )

db_pool: Optional[asyncpg.Pool] = None

async def get_db_pool():
    global db_pool
    if not db_pool and os.getenv("DATABASE_URL"):
        try:
            db_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))
            print("✅ Database pool created")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
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
    return {
        "message": "RAG Platform API v2.0",
        "status": "running",
        "features": {
            "vector_store": vector_store is not None,
            "database": db_pool is not None,
            "gemini_support": GEMINI_API_KEY is not None
        }
    }

@app.get("/health")
async def health_check():
    health = {
        "status": "healthy",
        "checks": {
            "supabase": False,
            "openai": False, 
            "database": False,
            "gemini": False
        }
    }
    
    try:
        if supabase:
            # Test Supabase connection
            supabase.table("documents").select("*").limit(1).execute()
            health["checks"]["supabase"] = True
    except:
        pass
        
    try:
        if embeddings:
            # Test OpenAI connection
            embeddings.embed_query("test")
            health["checks"]["openai"] = True
    except:
        pass
        
    try:
        if db_pool:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health["checks"]["database"] = True
    except:
        pass
        
    health["checks"]["gemini"] = bool(GEMINI_API_KEY)
    
    if not any(health["checks"].values()):
        health["status"] = "degraded"
        
    return health

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not vector_store:
        raise HTTPException(503, "Vector store not available. Check environment configuration.")
    
    # Validation
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    if file.size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(400, "File too large (max 50MB)")
    
    try:
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        
        if len(pdf_reader.pages) > 1000:
            raise HTTPException(400, "PDF too long (max 1000 pages)")
        
        # Extract text with page tracking
        text_parts = []
        page_map = {}
        
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    start_pos = len(" ".join(text_parts))
                    text_parts.append(f"[PAGE {i+1}]\n{page_text}")
                    page_map[start_pos] = i + 1
            except Exception as e:
                print(f"Warning: Could not extract text from page {i+1}: {e}")
                continue
        
        if not text_parts:
            raise HTTPException(400, "Could not extract any text from PDF")
        
        full_text = "\n\n".join(text_parts)
        
        # Intelligent text splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,  # Larger chunks for better context
            chunk_overlap=250, # Substantial overlap to maintain connections
            length_function=lambda t: len(tiktoken.encoding_for_model("gpt-4").encode(t)),
            separators=["\n\n[PAGE", "\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )
        
        chunks = splitter.split_text(full_text)
        
        # Create documents with rich metadata
        doc_id = clean_filename(file.filename)
        documents = []
        
        for i, chunk in enumerate(chunks):
            # Extract page number from chunk
            page_match = re.search(r'\[PAGE (\d+)\]', chunk)
            page_num = int(page_match.group(1)) if page_match else 1
            
            # Clean chunk (remove page markers for embedding)
            clean_chunk = re.sub(r'\[PAGE \d+\]\n?', '', chunk).strip()
            
            if clean_chunk:  # Only add non-empty chunks
                documents.append(Document(
                    page_content=clean_chunk,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "filename": file.filename,
                        "page": page_num,
                        "chunk_index": f"{i+1}/{len(chunks)}",
                        "total_pages": len(pdf_reader.pages),
                        "chunk_size": len(clean_chunk),
                        "has_page_marker": bool(page_match)
                    }
                ))
        
        if not documents:
            raise HTTPException(400, "No valid text chunks could be created")
        
        # Store in vector database
        vector_store.add_documents(documents)
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": len(pdf_reader.pages),
            "chunks_created": len(documents),
            "total_characters": len(full_text),
            "message": f"Successfully processed {len(pdf_reader.pages)} pages into {len(documents)} searchable chunks"
        }
        
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(400, "Invalid or corrupted PDF file")
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/docs")
async def list_documents():
    if not db_pool:
        return {"documents": [], "message": "Database not available"}
    
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT 
                    metadata->>'doc_id' as doc_id,
                    metadata->>'filename' as filename,
                    MAX((metadata->>'total_pages')::int) as total_pages,
                    COUNT(*) as chunk_count,
                    SUM((metadata->>'chunk_size')::int) as total_characters,
                    MIN(TO_TIMESTAMP((metadata->>'timestamp')::bigint)) as uploaded_at
                FROM documents 
                WHERE metadata->>'doc_id' IS NOT NULL
                GROUP BY metadata->>'doc_id', metadata->>'filename'
                ORDER BY uploaded_at DESC
            """)
        
        documents = []
        for row in rows:
            documents.append({
                "doc_id": row["doc_id"],
                "filename": row["filename"],
                "total_pages": row["total_pages"] or 0,
                "chunk_count": row["chunk_count"],
                "total_characters": row["total_characters"] or 0,
                "uploaded_at": row["uploaded_at"]
            })
        
        return {"documents": documents, "total": len(documents)}
        
    except Exception as e:
        print(f"List documents error: {e}")
        return {"documents": [], "error": str(e)}

@app.get("/ask")
async def ask_question(doc_id: str, q: str):
    if not vector_store:
        raise HTTPException(503, "Vector store not available")
    
    if not q.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    try:
        # Enhanced retrieval with multiple strategies
        results = vector_store.similarity_search(
            q,
            k=10,  # Get more results for better context
            filter={"doc_id": doc_id}
        )
        
        if not results:
            return {
                "answer": "No relevant information found in the document for this question.",
                "sources": [],
                "confidence": 0.0,
                "doc_id": doc_id
            }
        
        # Build rich context with page references
        context_parts = []
        sources = []
        page_set = set()
        
        for i, result in enumerate(results):
            page = result.metadata.get("page", "unknown")
            page_set.add(page)
            
            context_parts.append(f"[Source {i+1}, Page {page}]\n{result.page_content}")
            
            sources.append({
                "source_id": i + 1,
                "chunk_id": result.metadata.get("chunk_id", 0),
                "page": page,
                "text_preview": result.page_content[:200] + "..." if len(result.page_content) > 200 else result.page_content,
                "relevance_score": getattr(result, 'score', 0.8),
                "chunk_index": result.metadata.get("chunk_index", "unknown")
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate comprehensive answer
        llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        prompt = f"""Based on the following document excerpts, provide a comprehensive and accurate answer to the user's question.

DOCUMENT CONTEXT (from {len(page_set)} pages):
{context}

USER QUESTION: {q}

INSTRUCTIONS:
1. Provide a detailed, well-structured answer in English
2. Reference specific page numbers when citing information: [Page X]
3. If you find quantitative data, highlight it clearly
4. If you identify important entities (people, organizations, concepts), mention them
5. If the question cannot be fully answered from the provided context, state what information is missing
6. Organize your response with clear paragraphs
7. Be precise and avoid speculation beyond what's stated in the document

ANSWER:"""
        
        response = llm.invoke(prompt)
        
        # Calculate confidence based on result quality and coverage
        confidence = calculate_confidence(len(results), len(page_set), len(context))
        
        return {
            "answer": response.content,
            "sources": sources[:6],  # Return top 6 sources
            "confidence": confidence,
            "doc_id": doc_id,
            "pages_referenced": sorted(list(page_set)),
            "total_sources_found": len(results)
        }
        
    except Exception as e:
        print(f"Question answering error: {e}")
        raise HTTPException(500, f"Failed to process question: {str(e)}")

@app.get("/extract")
async def extract_knowledge(doc_id: str):
    if not vector_store:
        raise HTTPException(503, "Vector store not available")
    
    try:
        # Retrieve all document chunks
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT content, metadata 
                FROM documents 
                WHERE metadata->>'doc_id' = $1 
                ORDER BY (metadata->>'chunk_id')::int
            """, doc_id)
        
        if not rows:
            raise HTTPException(404, f"Document '{doc_id}' not found")
        
        # Reconstruct document with page markers
        text_parts = []
        doc_info = None
        
        for row in rows:
            if not doc_info:
                doc_info = row["metadata"]
            
            page = row["metadata"].get("page", 1)
            content = row["content"]
            
            # Add page marker if not present
            if f"[PAGE {page}]" not in content:
                text_parts.append(f"[PAGE {page}]\n{content}")
            else:
                text_parts.append(content)
        
        full_text = "\n\n".join(text_parts)
        token_count = len(tiktoken.encoding_for_model("gpt-4").encode(full_text))
        
        print(f"🔍 Starting extraction for {doc_id}: {token_count} tokens, {len(rows)} chunks")
        
        async def generate_extraction():
            try:
                # Choose extraction strategy based on document size
                if token_count > 120000:
                    print("📄 Large document detected - using sectioned processing")
                    extraction_data = await extract_large_document(rows, doc_id)
                elif token_count > 25000 and GEMINI_API_KEY:
                    print("🤖 Medium document - using Gemini 1.5 Pro")
                    extraction_data = await extract_with_gemini(full_text)
                else:
                    print("🧠 Standard document - using GPT-4o")
                    extraction_data = await extract_with_gpt4(full_text, token_count)
                
                # Add document metadata
                extraction_data["document_info"] = {
                    "doc_id": doc_id,
                    "filename": doc_info.get("filename", "unknown"),
                    "total_pages": doc_info.get("total_pages", len(set(r["metadata"].get("page", 1) for r in rows))),
                    "total_chunks": len(rows),
                    "total_tokens": token_count,
                    "extraction_method": extraction_data.get("model", "unknown"),
                    "processing_time": extraction_data.get("processing_time", "unknown")
                }
                
                # Validate and enhance
                extraction_data = validate_and_enhance_extraction(extraction_data)
                
                print(f"✅ Extraction completed for {doc_id}")
                
            except Exception as e:
                print(f"❌ Extraction failed for {doc_id}: {e}")
                extraction_data = create_error_response(doc_id, str(e), token_count)
            
            yield json.dumps(extraction_data, ensure_ascii=False, indent=2)
        
        return StreamingResponse(
            generate_extraction(), 
            media_type="application/json",
            headers={"Content-Disposition": f"inline; filename={doc_id}_extraction.json"}
        )
        
    except Exception as e:
        print(f"❌ Extract endpoint error: {e}")
        raise HTTPException(500, f"Extraction failed: {str(e)}")

async def extract_with_gpt4(full_text: str, token_count: int) -> Dict:
    """Extract using GPT-4o with intelligent text management"""
    import time
    start_time = time.time()
    
    try:
        # Handle very large documents by selecting key sections
        if token_count > 60000:
            print("📝 Large document - selecting key sections")
            full_text = select_key_sections(full_text, target_tokens=50000)
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0, request_timeout=300)
        prompt = EXTRACTION_PROMPT_V2.format(full_text=full_text)
        
        response = llm.invoke(prompt)
        
        # Parse and validate JSON
        extraction_data = parse_json_response(response.content)
        extraction_data["model"] = "gpt-4o"
        extraction_data["confidence"] = 0.90 if token_count < 30000 else 0.85
        extraction_data["processing_time"] = f"{time.time() - start_time:.1f}s"
        
        return extraction_data
        
    except Exception as e:
        print(f"GPT-4 extraction error: {e}")
        raise

async def extract_with_gemini(full_text: str) -> Dict:
    """Extract using Gemini 1.5 Pro for large documents"""
    import time
    start_time = time.time()
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = EXTRACTION_PROMPT_V2.format(full_text=full_text)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=8192
            )
        )
        
        # Clean and parse Gemini response
        response_text = response.text.strip()
        response_text = clean_gemini_response(response_text)
        
        extraction_data = parse_json_response(response_text)
        extraction_data["model"] = "gemini-1.5-pro"
        extraction_data["confidence"] = 0.95
        extraction_data["processing_time"] = f"{time.time() - start_time:.1f}s"
        
        return extraction_data
        
    except Exception as e:
        print(f"Gemini extraction error: {e}")
        # Fallback to GPT-4
        print("🔄 Falling back to GPT-4")
        return await extract_with_gpt4(full_text[:50000], 50000)

async def extract_large_document(rows: List, doc_id: str) -> Dict:
    """Process very large documents in intelligent sections"""
    import time
    start_time = time.time()
    
    print(f"📚 Processing large document with {len(rows)} chunks")
    
    # Group chunks into logical sections (by page ranges)
    sections = create_document_sections(rows, max_tokens_per_section=20000)
    print(f"📄 Created {len(sections)} sections")
    
    # Process each section
    all_quotes = []
    all_entities = []
    all_metrics = []
    all_relations = []
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    for i, section in enumerate(sections):
        print(f"🔍 Processing section {i+1}/{len(sections)}")
        
        section_text = "\n\n".join([chunk["content"] for chunk in section])
        pages = f"{section[0]['page']}-{section[-1]['page']}"
        
        # Custom prompt for sectioned processing
        section_prompt = f"""This is section {i+1} of {len(sections)} from a larger document (pages {pages}).

{EXTRACTION_PROMPT_V2.format(full_text=section_text)}

IMPORTANT: Use section-specific IDs with prefix s{i+1}_ (examples: s{i+1}_q1, s{i+1}_e1, s{i+1}_m1)
This ensures no ID conflicts when combining sections."""
        
        try:
            response = llm.invoke(section_prompt)
            section_data = parse_json_response(response.content)
            
            # Collect data from this section
            all_quotes.extend(section_data.get("quotes", []))
            all_entities.extend(section_data.get("entities", []))
            all_metrics.extend(section_data.get("metrics", []))
            all_relations.extend(section_data.get("relations", []))
            
        except Exception as e:
            print(f"⚠️ Section {i+1} processing failed: {e}")
            continue
    
    # Consolidate and deduplicate
    print("🔧 Consolidating extracted data")
    
    consolidated_data = {
        "quotes": all_quotes,
        "entities": deduplicate_entities(all_entities),
        "metrics": all_metrics,
        "relations": all_relations,
        "model": "gpt-4o-sectioned",
        "confidence": 0.80,
        "sections_processed": len(sections),
        "processing_time": f"{time.time() - start_time:.1f}s"
    }
    
    return consolidated_data

def create_document_sections(rows: List, max_tokens_per_section: int = 20000) -> List[List]:
    """Intelligently group chunks into sections"""
    sections = []
    current_section = []
    current_tokens = 0
    
    for row in rows:
        content = row["content"]
        chunk_tokens = len(tiktoken.encoding_for_model("gpt-4").encode(content))
        
        if current_tokens + chunk_tokens > max_tokens_per_section and current_section:
            sections.append(current_section)
            current_section = []
            current_tokens = 0
        
        current_section.append({
            "content": content,
            "page": row["metadata"].get("page", 1),
            "chunk_id": row["metadata"].get("chunk_id", 0)
        })
        current_tokens += chunk_tokens
    
    if current_section:
        sections.append(current_section)
    
    return sections

def deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """Intelligent entity deduplication"""
    if not entities:
        return []
    
    unique_entities = {}
    
    for entity in entities:
        name = entity.get("name", "").lower().strip()
        if not name:
            continue
        
        # Simple name matching - could be enhanced with fuzzy matching
        if name not in unique_entities:
            unique_entities[name] = entity
        else:
            # Merge references
            existing = unique_entities[name]
            existing["quote_ids"] = list(set(
                existing.get("quote_ids", []) + entity.get("quote_ids", [])
            ))
            existing["metric_ids"] = list(set(
                existing.get("metric_ids", []) + entity.get("metric_ids", [])
            ))
            
            # Keep the more detailed description
            if len(entity.get("description", "")) > len(existing.get("description", "")):
                existing["description"] = entity["description"]
    
    return list(unique_entities.values())

def select_key_sections(text: str, target_tokens: int = 50000) -> str:
    """Select key sections from very large documents"""
    sections = text.split("[PAGE")
    if len(sections) <= 3:
        return text
    
    # Take beginning, middle, and end sections
    total_sections = len(sections)
    section_size = target_tokens // 3
    
    start_sections = sections[:max(1, total_sections//4)]
    middle_start = total_sections//2 - total_sections//8
    middle_end = total_sections//2 + total_sections//8
    middle_sections = sections[middle_start:middle_end]
    end_sections = sections[-max(1, total_sections//4):]
    
    selected_text = "[PAGE".join(start_sections[:3] + ["...[CONTENT OMITTED]...", "[PAGE"] + middle_sections[:3] + ["...[CONTENT OMITTED]...", "[PAGE"] + end_sections[:3])
    
    return selected_text[:target_tokens * 4]  # Rough character limit

def clean_gemini_response(response: str) -> str:
    """Clean Gemini response to extract valid JSON"""
    # Remove markdown code blocks
    response = re.sub(r'^```json\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'```\s*$', '', response, flags=re.MULTILINE)
    
    # Find JSON object boundaries
    start = response.find('{')
    end = response.rfind('}') + 1
    
    if start != -1 and end > start:
        return response[start:end]
    
    return response

def parse_json_response(response_text: str) -> Dict:
    """Robust JSON parsing with error handling"""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        
        # Try to extract JSON from markdown or other formatting
        cleaned = clean_gemini_response(response_text)
        try:
            return json.loads(cleaned)
        except:
            # Return minimal structure if parsing completely fails
            return {
                "quotes": [],
                "entities": [],
                "metrics": [],
                "relations": [],
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
            }

def validate_and_enhance_extraction(data: Dict) -> Dict:
    """Comprehensive validation and enhancement of extracted data"""
    # Ensure all required fields exist
    data.setdefault("quotes", [])
    data.setdefault("entities", [])
    data.setdefault("metrics", [])
    data.setdefault("relations", [])
    
    # Validate and collect IDs
    quote_ids = set()
    entity_ids = set()
    metric_ids = set()
    
    # Process quotes
    for i, quote in enumerate(data["quotes"]):
        if not quote.get("id"):
            quote["id"] = f"q{i+1}"
        quote_ids.add(quote["id"])
        
        # Ensure required fields
        quote.setdefault("text", "")
        quote.setdefault("author", "unknown")
        quote.setdefault("entity_ids", [])
        quote.setdefault("metric_ids", [])
    
    # Process entities
    for i, entity in enumerate(data["entities"]):
        if not entity.get("id"):
            entity["id"] = f"e{i+1}"
        entity_ids.add(entity["id"])
        
        # Ensure required fields
        entity.setdefault("name", "unnamed")
        entity.setdefault("type", "concept")
        entity.setdefault("quote_ids", [])
        entity.setdefault("metric_ids", [])
    
    # Process metrics
    for i, metric in enumerate(data["metrics"]):
        if not metric.get("id"):
            metric["id"] = f"m{i+1}"
        metric_ids.add(metric["id"])
        
        # Ensure required fields
        metric.setdefault("value", "")
        metric.setdefault("context", "")
        metric.setdefault("entity_ids", [])
        metric.setdefault("quote_ids", [])
    
    # Validate cross-references
    validation_issues = []
    
    for quote in data["quotes"]:
        for eid in quote.get("entity_ids", []):
            if eid not in entity_ids:
                validation_issues.append(f"Quote {quote['id']} references non-existent entity {eid}")
    
    # Generate comprehensive statistics
    data["extraction_stats"] = {
        "totals": {
            "quotes": len(data["quotes"]),
            "entities": len(data["entities"]),
            "metrics": len(data["metrics"]),
            "relations": len(data["relations"])
        },
        "cross_references": {
            "quotes_with_entities": len([q for q in data["quotes"] if q.get("entity_ids")]),
            "quotes_with_metrics": len([q for q in data["quotes"] if q.get("metric_ids")]),
            "entities_with_quotes": len([e for e in data["entities"] if e.get("quote_ids")]),
            "entities_with_metrics": len([e for e in data["entities"] if e.get("metric_ids")])
        },
        "quality_indicators": {
            "has_quotes": len(data["quotes"]) > 0,
            "has_entities": len(data["entities"]) > 0,
            "has_metrics": len(data["metrics"]) > 0,
            "has_relations": len(data["relations"]) > 0,
            "well_connected": len([item for collection in [data["quotes"], data["entities"], data["metrics"]] 
                                  for item in collection 
                                  if any([item.get("quote_ids"), item.get("entity_ids"), item.get("metric_ids")])]) > 0
        },
        "validation_issues": validation_issues
    }
    
    # Calculate extraction quality score
    quality_score = 0
    if data["extraction_stats"]["totals"]["quotes"] > 0: quality_score += 0.25
    if data["extraction_stats"]["totals"]["entities"] > 0: quality_score += 0.25
    if data["extraction_stats"]["totals"]["metrics"] > 0: quality_score += 0.25
    if data["extraction_stats"]["quality_indicators"]["well_connected"]: quality_score += 0.25
    
    data["extraction_quality"] = round(quality_score, 2)
    
    return data

def create_error_response(doc_id: str, error_message: str, token_count: int) -> Dict:
    """Create standardized error response"""
    return {
        "quotes": [],
        "entities": [],
        "metrics": [],
        "relations": [],
        "extraction_stats": {
            "totals": {"quotes": 0, "entities": 0, "metrics": 0, "relations": 0},
            "error": True
        },
        "extraction_quality": 0.0,
        "confidence": 0.0,
        "error": error_message,
        "document_info": {
            "doc_id": doc_id,
            "total_tokens": token_count,
            "extraction_method": "failed"
        }
    }

def calculate_confidence(result_count: int, page_count: int, context_length: int) -> float:
    """Calculate confidence score for Q&A"""
    base_confidence = 0.6
    
    # More results = higher confidence
    if result_count >= 8: base_confidence += 0.15
    elif result_count >= 5: base_confidence += 0.10
    elif result_count >= 3: base_confidence += 0.05
    
    # More pages = better coverage
    if page_count >= 5: base_confidence += 0.10
    elif page_count >= 3: base_confidence += 0.05
    
    # Longer context = more comprehensive
    if context_length >= 5000: base_confidence += 0.10
    elif context_length >= 2000: base_confidence += 0.05
    
    return min(0.95, base_confidence)

def clean_filename(filename: str) -> str:
    """Clean filename to create valid doc_id"""
    # Remove extension and clean
    name = os.path.splitext(filename)[0]
    # Replace problematic characters
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Remove multiple underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name if name else "document"

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Platform API v2.0")
    uvicorn.run(app, host="0.0.0.0", port=8000)