-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    text TEXT NOT NULL,
    total_pages INTEGER NOT NULL,
    total_characters INTEGER NOT NULL,
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create document embeddings table for vector search
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX ON document_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Add RLS policies
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;

-- Allow anonymous access (for simplicity - adjust for production)
CREATE POLICY "Enable read access for all users" ON documents
    FOR SELECT USING (true);

CREATE POLICY "Enable insert for all users" ON documents
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable delete for all users" ON documents
    FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON document_embeddings
    FOR SELECT USING (true);

CREATE POLICY "Enable insert for all users" ON document_embeddings
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Enable delete for all users" ON document_embeddings
    FOR DELETE USING (true);