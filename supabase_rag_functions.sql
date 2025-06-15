-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Update document_embeddings table to ensure proper structure
ALTER TABLE document_embeddings 
ADD COLUMN IF NOT EXISTS similarity FLOAT;

-- Create the vector similarity search function
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding vector(1536),
  filter_doc_id text DEFAULT NULL,
  match_threshold float DEFAULT 0.3,
  match_count int DEFAULT 6
)
RETURNS TABLE (
  id uuid,
  doc_id uuid,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN query
  SELECT
    document_embeddings.id,
    document_embeddings.doc_id,
    document_embeddings.content,
    document_embeddings.metadata,
    1 - (document_embeddings.embedding <=> query_embedding) as similarity
  FROM document_embeddings
  WHERE 
    (filter_doc_id IS NULL OR document_embeddings.metadata->>'doc_id' = filter_doc_id)
    AND 1 - (document_embeddings.embedding <=> query_embedding) > match_threshold
  ORDER BY document_embeddings.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Create function to initialize embeddings table if needed
CREATE OR REPLACE FUNCTION create_embeddings_table_if_not_exists()
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
  -- Table should already exist, this is just a placeholder
  -- to ensure the function exists for error handling
  NULL;
END;
$$;

-- Ensure proper indexes exist
DROP INDEX IF EXISTS document_embeddings_embedding_idx;
CREATE INDEX document_embeddings_embedding_idx 
ON document_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create index on metadata for faster filtering
CREATE INDEX IF NOT EXISTS document_embeddings_metadata_doc_id_idx 
ON document_embeddings USING btree ((metadata->>'doc_id'));

-- Update RLS policies to be more permissive for now
DROP POLICY IF EXISTS "Enable read access for all users" ON document_embeddings;
DROP POLICY IF EXISTS "Enable insert for all users" ON document_embeddings;
DROP POLICY IF EXISTS "Enable delete for all users" ON document_embeddings;

CREATE POLICY "Enable full access for embeddings" ON document_embeddings
FOR ALL USING (true) WITH CHECK (true);