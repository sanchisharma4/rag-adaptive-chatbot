
import sqlite3
from datetime import datetime
import os
import math
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader
import chromadb


# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load API key from .env
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")

# Connect to Gemini
client = genai.Client(api_key=API_KEY) if API_KEY else None


# Gemini embedding helper
def get_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    if client is None:
        raise RuntimeError("Gemini API key is not configured.")

    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []

    # Process texts in batches
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]

        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768
            )
        )

        all_embeddings.extend(
            [item.values for item in response.embeddings]
        )

    return all_embeddings


# Calculate cosine similarity
def cosine_similarity_scores(query_embedding, embeddings):
    query_norm = math.sqrt(
        sum(value * value for value in query_embedding)
    )

    scores = []

    for embedding in embeddings:
        embedding_norm = math.sqrt(
            sum(value * value for value in embedding)
        )

        if query_norm == 0 or embedding_norm == 0:
            scores.append(0.0)
            continue

        dot_product = sum(
            a * b for a, b in zip(query_embedding, embedding)
        )

        scores.append(
            dot_product / (query_norm * embedding_norm)
        )

    return scores


# Project storage folders
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# Initialize ChromaDB with Gemini embeddings
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="academic_documents_gemini"
)


# SQLite Adaptive Memory Database
DB_PATH = BASE_DIR / "memory.db"


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_memory():
    connection = get_db_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_text TEXT NOT NULL,
            memory_type TEXT DEFAULT 'episodic',
            created_at TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS pending_memory_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            old_memory_text TEXT NOT NULL,
            new_memory_text TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


initialize_memory()


# Save memory
def save_memory(memory_text, memory_type="episodic"):
    connection = get_db_connection()

    memory_text = memory_text.strip()

    existing_memory = connection.execute(
        """
        SELECT id FROM memories
        WHERE LOWER(TRIM(memory_text)) = LOWER(?)
        AND memory_type = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (memory_text, memory_type)
    ).fetchone()

    if existing_memory:
        connection.execute(
            """
            UPDATE memories
            SET memory_text = ?, created_at = ?
            WHERE id = ?
            """,
            (
                memory_text,
                datetime.now().isoformat(),
                existing_memory["id"]
            )
        )

        print("Existing memory updated.")

    else:
        connection.execute(
            """
            INSERT INTO memories (memory_text, memory_type, created_at)
            VALUES (?, ?, ?)
            """,
            (
                memory_text,
                memory_type,
                datetime.now().isoformat()
            )
        )

        print("New memory saved.")

    connection.commit()
    connection.close()


# Retrieve saved memories
def get_memories(limit=10):
    connection = get_db_connection()

    rows = connection.execute(
        """
        SELECT memory_text, memory_type, created_at
        FROM memories
        ORDER BY
            CASE
                WHEN memory_type = 'semantic' THEN 0
                ELSE 1
            END,
            id DESC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    connection.close()

    return [dict(row) for row in rows]


# Phase 5: Find Similar Memories
def find_similar_memory(
    memory_text,
    memory_type="semantic",
    threshold=0.80
):
    memories = get_memories(limit=100)

    memories = [
        memory for memory in memories
        if memory["memory_type"] == memory_type
    ]

    if not memories:
        return None

    memory_texts = [
        memory["memory_text"] for memory in memories
    ]

    new_embedding = get_embeddings(
        memory_text,
        task_type="SEMANTIC_SIMILARITY"
    )[0]

    existing_embeddings = get_embeddings(
        memory_texts,
        task_type="SEMANTIC_SIMILARITY"
    )

    similarities = cosine_similarity_scores(
        new_embedding,
        existing_embeddings
    )

    best_index = max(
        range(len(similarities)),
        key=lambda i: similarities[i]
    )

    best_score = similarities[best_index]

    if best_score >= threshold:
        return {
            "memory_text": memory_texts[best_index],
            "similarity": round(best_score, 3)
        }

    return None


# Create pending memory update
def create_pending_memory_update(
    old_memory_text,
    new_memory_text,
    memory_type="semantic"
):
    connection = get_db_connection()

    cursor = connection.execute(
        """
        INSERT INTO pending_memory_updates
        (old_memory_text, new_memory_text, memory_type, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            old_memory_text,
            new_memory_text,
            memory_type,
            datetime.now().isoformat()
        )
    )

    update_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return update_id


# Phase 5: Confirm or Reject Memory Update
def resolve_pending_memory_update(update_id, approved):
    connection = get_db_connection()

    pending = connection.execute(
        """
        SELECT * FROM pending_memory_updates
        WHERE id = ?
        """,
        (update_id,)
    ).fetchone()

    if not pending:
        connection.close()
        return "Pending memory update not found."

    if approved:
        connection.execute(
            """
            UPDATE memories
            SET memory_text = ?, created_at = ?
            WHERE id = (
                SELECT id FROM memories
                WHERE memory_text = ? AND memory_type = ?
                ORDER BY id DESC
                LIMIT 1
            )
            """,
            (
                pending["new_memory_text"],
                datetime.now().isoformat(),
                pending["old_memory_text"],
                pending["memory_type"]
            )
        )

    connection.execute(
        "DELETE FROM pending_memory_updates WHERE id = ?",
        (update_id,)
    )

    connection.commit()
    connection.close()

    if approved:
        return "Memory successfully updated."

    return "Previous memory kept unchanged."


# Phase 4: Adaptive Memory Retrieval
def retrieve_relevant_memories(
    query,
    limit=5,
    threshold=0.35
):
    memories = get_memories(limit=100)

    if not memories:
        return []

    # Exclude question-like memories
    memories = [
        memory for memory in memories
        if not memory["memory_text"].strip().endswith("?")
    ]

    if not memories:
        return []

    memory_texts = [
        memory["memory_text"] for memory in memories
    ]

    query_embedding = get_embeddings(
        query,
        task_type="SEMANTIC_SIMILARITY"
    )[0]

    memory_embeddings = get_embeddings(
        memory_texts,
        task_type="SEMANTIC_SIMILARITY"
    )

    similarities = cosine_similarity_scores(
        query_embedding,
        memory_embeddings
    )

    relevant_memories = []

    for index, score in enumerate(similarities):
        if score >= threshold:
            relevant_memories.append({
                "memory_text": memory_texts[index],
                "memory_type": memories[index]["memory_type"],
                "similarity": round(score, 3)
            })

    relevant_memories.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return relevant_memories[:limit]


# Classify memory
def classify_memory(message):
    message = message.strip().lower()

    if len(message) < 8:
        return None

    greetings = [
        "hello", "hi", "hey",
        "good morning", "good night",
        "thanks", "thank you"
    ]

    if message in greetings:
        return None

    question_patterns = [
        "which is my",
        "what is my",
        "what's my",
        "do you remember",
        "can you tell me my",
        "who am i"
    ]

    for pattern in question_patterns:
        if pattern in message:
            return None

    semantic_patterns = [
        "my favourite", "my favorite",
        "i prefer", "i like",
        "i dislike", "i don't like",
        "i am", "i'm", "my goal",
        "i want to", "i study",
        "my interest"
    ]

    for pattern in semantic_patterns:
        if pattern in message:
            return "semantic"

    if message.startswith((
        "prefer ", "like ", "dislike ", "don't like "
    )):
        return "semantic"

    return None


# Create FastAPI application
app = FastAPI(title="RAG Adaptive Chatbot")


# Serve frontend files
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = []


class MemoryUpdateRequest(BaseModel):
    update_id: int
    approved: bool


# PDF Upload Endpoint
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed."}

    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}.pdf"
    file_path = UPLOAD_DIR / safe_filename

    try:
        content = await file.read()

        if not content:
            return {"error": "The uploaded PDF is empty."}

        with open(file_path, "wb") as f:
            f.write(content)

        reader = PdfReader(str(file_path))

        chunks = []

        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""

            words = page_text.split()

            for i in range(0, len(words), 150):
                chunk = " ".join(words[i:i + 150])

                if chunk.strip():
                    chunks.append({
                        "text": chunk,
                        "page": page_number + 1
                    })

        if not chunks:
            file_path.unlink(missing_ok=True)

            return {
                "error": "No readable text found. This PDF may be scanned."
            }

        # Generate Gemini document embeddings
        texts = [
            chunk["text"] for chunk in chunks
        ]

        embeddings = get_embeddings(
            texts,
            task_type="RETRIEVAL_DOCUMENT"
        )

        # Store documents in ChromaDB
        collection.add(
            ids=[
                f"{file_id}_{i}"
                for i in range(len(chunks))
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "filename": file.filename,
                    "file_id": file_id,
                    "page": chunk["page"]
                }
                for chunk in chunks
            ]
        )

        return {
            "message": "PDF uploaded and processed successfully!",
            "filename": file.filename,
            "chunks_created": len(chunks)
        }

    except Exception as e:
        print("PDF Upload Error:", e)
        file_path.unlink(missing_ok=True)

        return {
            "error": "Unable to process this PDF. Check the terminal."
        }


# Home Endpoint
@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "static" / "index.html"
    )


# Chat Endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    user_message = request.message.strip()

    if not user_message:
        return {"reply": "Please enter a message."}

    if not client:
        return {
            "reply": "API key not found. Please check your .env file."
        }

    try:
        # Retrieve relevant memories
        memories = retrieve_relevant_memories(
            user_message,
            limit=5,
            threshold=0.35
        )

        memory_context = "\n".join(
            [
                f"{memory['memory_type']}: {memory['memory_text']}"
                for memory in memories
            ]
        )

        # Search uploaded documents using Gemini embeddings
        query_embedding = get_embeddings(
            user_message,
            task_type="RETRIEVAL_QUERY"
        )

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        retrieved_chunks = results.get(
            "documents", [[]]
        )[0]

        if retrieved_chunks:
            context = "\n\n".join(retrieved_chunks)
        else:
            context = "No relevant uploaded document found."

        # Prepare recent conversation history
        history_lines = []

        for item in request.conversation_history[-10:]:
            role = item.get("role")
            content = item.get("content", "")

            if (
                role in ["user", "assistant"]
                and isinstance(content, str)
            ):
                history_lines.append(
                    f"{role}: {content[:2000]}"
                )

        conversation_context = "\n".join(history_lines)

        print(
            "DEBUG - Conversation context:",
            conversation_context
        )

        # Create prompt
        prompt = f"""
        You are a helpful academic assistant with memory.

        PREVIOUS USER MEMORIES:
        {memory_context}

        RECENT CONVERSATION HISTORY:
        {conversation_context}

        DOCUMENT CONTEXT:
        {context}

        USER QUESTION:
        {user_message}

        Use the recent conversation history to understand the user's
        current topic and follow-up questions.

        Resolve words such as "it", "its", "they", and "their"
        using the most relevant previous conversation.

        Do not change the topic to an unrelated subject from the
        uploaded document.

        Use document context only when it is relevant to the
        current question.

        For general academic questions, you may use your general
        knowledge when document information is unavailable.
        Clearly distinguish general knowledge from document-based
        information.

        Do not invent facts.
        """

        # Classify and save current user message
        memory_type = classify_memory(user_message)

        print(
            "DEBUG - User message:",
            repr(user_message)
        )

        print(
            "DEBUG - Memory type:",
            memory_type
        )

        if memory_type is not None:

            similar_memory = find_similar_memory(
                user_message,
                memory_type
            )

            if (
                similar_memory
                and similar_memory["memory_text"].strip().lower()
                != user_message.strip().lower()
            ):

                update_id = create_pending_memory_update(
                    similar_memory["memory_text"],
                    user_message,
                    memory_type
                )

                return {
                    "reply": (
                        f"You previously told me: "
                        f"{similar_memory['memory_text']}\n\n"
                        f"Would you like to change it to: "
                        f"{user_message}?"
                    ),
                    "pending_update_id": update_id,
                    "requires_confirmation": True
                }

            save_memory(
                user_message,
                memory_type=memory_type
            )

        # Generate Gemini response
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        return {
            "reply": response.text or "No response generated."
        }

    except Exception as e:
        print("Chat Error:", e)

        return {
            "reply": "Sorry, something went wrong. Check the terminal."
        }


# Confirm Memory Update Endpoint
@app.post("/memory/confirm")
def confirm_memory_update(request: MemoryUpdateRequest):

    result = resolve_pending_memory_update(
        request.update_id,
        request.approved
    )

    return {
        "message": result
    }
