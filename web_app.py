from flask import Flask, Response, render_template, request, jsonify
import chromadb
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

import hashlib

import retrieval
from mistral_client import (
    CHAT_MODEL,
    MistralEmbedding,
    chat as mistral_chat,
    chat_stream as mistral_chat_stream,
)
from prompts import build_system_prompt, build_user_prompt, describe_corpus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration
COLLECTION_NAME = "valorant_patches"
PATCH_NOTES_FILE = "feedme_patchnotes.json"
# Persist the index so a restart does not re-embed the whole corpus. Set
# CHROMA_PATH empty to force the in-memory client.
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")

def setup_chromadb():
    try:
        if CHROMA_PATH:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            logger.info(f"ChromaDB persisting to {CHROMA_PATH}")
        else:
            client = chromadb.Client()
            logger.info("ChromaDB running in memory")

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=MistralEmbedding()
        )
        return client, collection
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {e}")
        return None, None

dataset = []
latest_patch = None
oldest_patch = None
client = None
collection = None
patch_keys = []


def corpus_fingerprint(dataset: List[Dict[str, Any]]) -> str:
    """Identity of the indexed corpus, so a stale index gets rebuilt.

    Covers the chunking parameters as well as the data: changing how notes are
    split has to invalidate an index built under the old scheme.
    """
    h = hashlib.sha256()
    h.update(f"v2:{retrieval.CHUNK_MAX_CHARS}:{retrieval.CHUNK_MIN_CHARS}".encode())
    for record in dataset:
        h.update((record.get("title") or "").encode("utf-8"))
        h.update(str(len(record.get("final_content") or "")).encode())
    return h.hexdigest()[:16]

def load_patch_notes(filename: str) -> List[Dict[str, Any]]:
    global dataset, latest_patch, oldest_patch
    
    try:
        if not os.path.exists(filename):
            logger.error(f"File '{filename}' not found!")
            return []

        with open(filename, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # sort then use later 
        dataset_sorted = sorted(
            dataset,
            key=lambda x: datetime.strptime(x.get("published", "1970-01-01"), "%Y-%m-%d")
        )

        # oldest and latest patches
        if dataset_sorted:
            oldest_patch = dataset_sorted[0]
            latest_patch = dataset_sorted[-1]
            logger.info(f"Loaded {len(dataset_sorted)} patches. "
                       f"Range: {oldest_patch.get('title', 'N/A')} to {latest_patch.get('title', 'N/A')}")

        return dataset_sorted
    except Exception as e:
        logger.error(f"Error loading patch notes: {e}")
        return []

def add_documents_to_collection(collection, dataset: List[Dict[str, Any]]) -> bool:
    if not dataset:
        logger.warning("No data to add to collection")
        return False
    
    try:
        documents, metadatas, ids = retrieval.build_chunks(dataset)

        if not documents:
            logger.warning("No valid documents found to add")
            return False

        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )

        logger.info(f"Indexed {len(documents)} chunks from {len(dataset)} patch notes")
        return True
    except Exception as e:
        logger.error(f"Error adding documents to collection: {e}")
        return False

def validate_message(message: str) -> Optional[str]:
    """Validate user input message."""
    if not message or not message.strip():
        return "Message cannot be empty"
    
    if len(message) > 1000:  
        return "Message too long (max 1000 characters)"
    
    # Basic sanitization para di ma hack
    dangerous_chars = ['<script', '</script', 'javascript:', 'data:']
    message_lower = message.lower()
    for char in dangerous_chars:
        if char in message_lower:
            return "Message contains invalid characters"
    
    return None

def chat_chris(prompt: str, collection, k: int = 5) -> Dict[str, Any]:
    """Answer a question and report which patches the answer was drawn from."""
    try:
        validation_error = validate_message(prompt)
        if validation_error:
            return {"answer": f"Error: {validation_error}", "sources": []}

        result = retrieval.search(
            collection,
            prompt,
            known_keys=patch_keys,
            latest_key=(latest_patch or {}).get("patch_key"),
            k=k,
        )
        rows = result["rows"]
        context = retrieval.format_context(rows)

        sources = retrieval.cited_sources(rows)
        logger.info(
            f"retrieval mode={result['mode']} chunks={len(rows)} "
            f"patches={[s['title'] for s in sources][:4]}"
        )

        corpus = describe_corpus(dataset)
        system_prompt = build_system_prompt(corpus)
        user_prompt = build_user_prompt(context, prompt)

        answer = mistral_chat(system_prompt, user_prompt)
        return {"answer": answer, "sources": sources}

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "answer": "Sorry, I'm having trouble processing your request right now. "
                      "Try asking again in a moment!",
            "sources": [],
        }

def initialize_system():
    global client, collection, patch_keys

    logger.info("Initializing Valorant Patch Notes RAG System...")

    client, collection = setup_chromadb()
    if not collection:
        logger.error("Failed to setup ChromaDB.")
        return False

    dataset_sorted = load_patch_notes(PATCH_NOTES_FILE)
    if not dataset_sorted:
        logger.error("No patch notes loaded.")
        return False

    patch_keys = [r["patch_key"] for r in dataset_sorted if r.get("patch_key")]

    fingerprint = corpus_fingerprint(dataset_sorted)
    indexed = (collection.metadata or {}).get("fingerprint")

    if collection.count() > 0 and indexed == fingerprint:
        logger.info(f"Reusing index of {collection.count()} chunks ({fingerprint})")
        logger.info("System initialization complete!")
        return True

    if collection.count() > 0 or indexed:
        logger.info(f"Index is stale or incomplete ({indexed} != {fingerprint}); rebuilding")
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=MistralEmbedding(),
        )

    if not add_documents_to_collection(collection, dataset_sorted):
        logger.error("Failed to add documents; index left unstamped so the next "
                     "start rebuilds it rather than trusting a partial index.")
        return False

    # Stamped only after a complete build. A run that dies partway through
    # leaves no fingerprint, so the next start rebuilds instead of serving
    # an index that is silently missing chunks.
    collection.modify(metadata={"fingerprint": fingerprint})
    logger.info("System initialization complete!")
    return True

initialize_system()

@app.route('/')
def index():
    corpus = describe_corpus(dataset)
    return render_template(
        'index.html',
        patch_count=len(dataset),
        latest_patch=corpus.latest_title.replace("VALORANT Patch Notes ", ""),
        latest_date=corpus.latest_date,
        oldest_date=corpus.oldest_date,
    )

@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        
        validation_error = validate_message(user_message)
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        if not collection:
            return jsonify({'error': 'System not initialized properly'}), 500
        
        result = chat_chris(user_message, collection)

        user_msg = {
            'id': str(uuid.uuid4()),
            'message': user_message,
            'sender': 'user',
            'timestamp': datetime.now().isoformat()
        }

        bot_msg = {
            'id': str(uuid.uuid4()),
            'message': result['answer'],
            'sources': result['sources'],
            'sender': 'bot',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'user_message': user_msg,
            'bot_message': bot_msg
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.route('/chat/stream', methods=['POST'])
def chat_stream_route():
    """Stream a reply as it is generated.

    Retrieval runs before the response opens, so a failure there is still a
    normal HTTP error rather than an error buried inside a 200 stream. Sources
    are known at that point too, and are sent as the first event.
    """
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    user_message = (request.get_json() or {}).get('message', '').strip()
    validation_error = validate_message(user_message)
    if validation_error:
        return jsonify({'error': validation_error}), 400
    if not collection:
        return jsonify({'error': 'System not initialized properly'}), 500

    try:
        result = retrieval.search(
            collection,
            user_message,
            known_keys=patch_keys,
            latest_key=(latest_patch or {}).get("patch_key"),
            k=5,
        )
        rows = result["rows"]
        sources = retrieval.cited_sources(rows)
        logger.info(f"retrieval mode={result['mode']} chunks={len(rows)} (stream)")

        corpus = describe_corpus(dataset)
        system_prompt = build_system_prompt(corpus)
        user_prompt = build_user_prompt(retrieval.format_context(rows), user_message)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return jsonify({'error': 'An internal error occurred'}), 500

    def events():
        yield _sse({'type': 'sources', 'sources': sources})
        try:
            for piece in mistral_chat_stream(system_prompt, user_prompt):
                yield _sse({'type': 'delta', 'text': piece})
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield _sse({'type': 'error',
                        'message': "Sorry, I lost my train of thought. Try that again?"})
        yield _sse({'type': 'done'})

    return Response(
        events(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            # Stops nginx-style proxies buffering the stream into one blob.
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/health')
def health():
    try:
        status = {
            'status': 'healthy',
            'collection_count': collection.count() if collection else 0,
            'has_latest_patch': latest_patch is not None,
            'has_oldest_patch': oldest_patch is not None,
            'total_patches': len(dataset)
        }
        
        if latest_patch:
            status['latest_patch'] = {
                'title': latest_patch.get('title', 'N/A'),
                'published': latest_patch.get('published', 'N/A')
            }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/stats')
def stats():
    try:
        if not collection:
            return jsonify({'error': 'System not initialized'}), 500
        
        return jsonify({
            'total_documents': collection.count(),
            'total_patches': len(dataset),
            'latest_patch': latest_patch.get('title', 'N/A') if latest_patch else 'N/A',
            'oldest_patch': oldest_patch.get('title', 'N/A') if oldest_patch else 'N/A',
            'date_range': {
                'oldest': oldest_patch.get('published', 'N/A') if oldest_patch else 'N/A',
                'latest': latest_patch.get('published', 'N/A') if latest_patch else 'N/A'
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to get stats'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)