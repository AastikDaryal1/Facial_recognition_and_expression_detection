-- scripts/init_db.sql
-- This file runs automatically on first postgres container start
-- (placed in /docker-entrypoint-initdb.d/)
-- It will NOT re-run on subsequent starts — only when postgres_data volume is fresh.

-- Enable the pgvector extension in the face_emotion_db database.
-- This adds the VECTOR column type used by face_embeddings.embedding.
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify it loaded correctly (shows in container logs on startup)
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';