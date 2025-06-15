-- ============================================
-- FIX RLS POLICIES FOR RAG PLATFORM
-- ============================================

-- 1. First, disable RLS temporarily to see if that's the issue
ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings DISABLE ROW LEVEL SECURITY;

-- 2. Drop all existing policies to start fresh
DROP POLICY IF EXISTS "Enable read access for all users" ON documents;
DROP POLICY IF EXISTS "Enable insert for all users" ON documents;
DROP POLICY IF EXISTS "Enable update for all users" ON documents;
DROP POLICY IF EXISTS "Enable delete for all users" ON documents;

DROP POLICY IF EXISTS "Enable read access for all users" ON document_embeddings;
DROP POLICY IF EXISTS "Enable insert for all users" ON document_embeddings;
DROP POLICY IF EXISTS "Enable update for all users" ON document_embeddings;
DROP POLICY IF EXISTS "Enable delete for all users" ON document_embeddings;

-- 3. Re-enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;

-- 4. Create new permissive policies for anonymous access
-- For documents table
CREATE POLICY "Allow anonymous read" ON documents
    FOR SELECT TO anon
    USING (true);

CREATE POLICY "Allow anonymous insert" ON documents
    FOR INSERT TO anon
    WITH CHECK (true);

CREATE POLICY "Allow anonymous update" ON documents
    FOR UPDATE TO anon
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow anonymous delete" ON documents
    FOR DELETE TO anon
    USING (true);

-- For document_embeddings table
CREATE POLICY "Allow anonymous read" ON document_embeddings
    FOR SELECT TO anon
    USING (true);

CREATE POLICY "Allow anonymous insert" ON document_embeddings
    FOR INSERT TO anon
    WITH CHECK (true);

CREATE POLICY "Allow anonymous update" ON document_embeddings
    FOR UPDATE TO anon
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow anonymous delete" ON document_embeddings
    FOR DELETE TO anon
    USING (true);

-- 5. Also grant permissions to authenticated users
-- For documents table
CREATE POLICY "Allow authenticated read" ON documents
    FOR SELECT TO authenticated
    USING (true);

CREATE POLICY "Allow authenticated insert" ON documents
    FOR INSERT TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow authenticated update" ON documents
    FOR UPDATE TO authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow authenticated delete" ON documents
    FOR DELETE TO authenticated
    USING (true);

-- For document_embeddings table
CREATE POLICY "Allow authenticated read" ON document_embeddings
    FOR SELECT TO authenticated
    USING (true);

CREATE POLICY "Allow authenticated insert" ON document_embeddings
    FOR INSERT TO authenticated
    WITH CHECK (true);

CREATE POLICY "Allow authenticated update" ON document_embeddings
    FOR UPDATE TO authenticated
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow authenticated delete" ON document_embeddings
    FOR DELETE TO authenticated
    USING (true);

-- 6. Grant table permissions
GRANT ALL ON documents TO anon;
GRANT ALL ON documents TO authenticated;
GRANT ALL ON document_embeddings TO anon;
GRANT ALL ON document_embeddings TO authenticated;

-- 7. Test query to verify access
-- This should return all documents
SELECT COUNT(*) as total_documents FROM documents;

-- 8. If you want to completely disable RLS (not recommended for production)
-- Uncomment these lines:
-- ALTER TABLE documents DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE document_embeddings DISABLE ROW LEVEL SECURITY;