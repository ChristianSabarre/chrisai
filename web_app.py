from flask import Flask, render_template, request, jsonify
import chromadb
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

from mistral_client import CHAT_MODEL, MistralEmbedding, chat as mistral_chat
from prompts import build_system_prompt, build_user_prompt, describe_corpus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration
COLLECTION_NAME = "valorant_patches"
PATCH_NOTES_FILE = "feedme_patchnotes.json"

def setup_chromadb():
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=MistralEmbedding()
        )
        logger.info("ChromaDB setup successful")
        return client, collection
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {e}")
        return None, None

dataset = []
latest_patch = None
oldest_patch = None
client = None
collection = None

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
        documents = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(dataset):
            content = doc.get("final_content", "")
            patch_number = doc.get("patch", f"patch_{i}")
            title = doc.get("title", f"Patch {patch_number}")
            
            if content.strip(): 
                documents.append(content)
                metadatas.append({
                    "source": "valorant_patch_notes",
                    "patch": patch_number,
                    "title": title,
                    "published": doc.get("published", ""),
                    "index": i
                })
                ids.append(f"patch_{patch_number}_{i}")

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
        
        logger.info(f"Successfully added {len(documents)} patch notes to ChromaDB collection!")
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

def chat_chris(prompt: str, collection, k: int = 5) -> str:
    try:
        validation_error = validate_message(prompt)
        if validation_error:
            return f"Error: {validation_error}"
        
        results = collection.query(
            query_texts=[prompt],
            n_results=k
        )
        
        if not results['documents'] or not results['documents'][0]:
            context = "No relevant patch notes found for your query."
            retrieved_docs = []
            retrieved_metadata = []
        else:
            retrieved_docs = results['documents'][0]
            retrieved_metadata = results['metadatas'][0] if results['metadatas'] else []

        latest_title = latest_patch.get("title", "N/A") if latest_patch else "N/A"
        latest_date = latest_patch.get("published", "N/A") if latest_patch else "N/A"

        enhanced_context_parts = []
        for i, doc in enumerate(retrieved_docs):
            metadata = retrieved_metadata[i] if i < len(retrieved_metadata) else {}
            patch_title = metadata.get('title', 'Unknown Patch')
            patch_date = metadata.get('published', 'Unknown Date')
            
            enhanced_context_parts.append(f"[PATCH: {patch_title} - Published: {patch_date}]\n{doc}")

        if latest_patch:
            latest_content = latest_patch.get("final_content", "")
            if latest_content and latest_content not in retrieved_docs:
                enhanced_context_parts.insert(0, f"[LATEST PATCH: {latest_title} - Published: {latest_date}]\n{latest_content}")

        context = "\n\n---\n\n".join(enhanced_context_parts) if enhanced_context_parts else "No relevant patch notes found."

        corpus = describe_corpus(dataset)
        system_prompt = build_system_prompt(corpus)
        user_prompt = build_user_prompt(context, prompt)

        return mistral_chat(system_prompt, user_prompt)

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I'm having trouble processing your request right now. Try asking again in a moment!"

def initialize_system():
    global client, collection
    
    logger.info("Initializing Valorant Patch Notes RAG System...")
    
    client, collection = setup_chromadb()
    if not collection:
        logger.error("Failed to setup ChromaDB.")
        return False
    
    dataset_sorted = load_patch_notes(PATCH_NOTES_FILE)
    if not dataset_sorted:
        logger.error("No patch notes loaded.")
        return False
    
    if collection.count() == 0:
        logger.info("Collection is empty. Adding patch notes...")
        if not add_documents_to_collection(collection, dataset_sorted):
            logger.error("Failed to add documents.")
            return False
    else:
        logger.info(f"Collection already contains {collection.count()} documents")
    
    logger.info("System initialization complete!")
    return True

initialize_system()

@app.route('/')
def index():
    return render_template('index.html')

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
        
        response = chat_chris(user_message, collection)
        
        user_msg = {
            'id': str(uuid.uuid4()),
            'message': user_message,
            'sender': 'user',
            'timestamp': datetime.now().isoformat()
        }
        
        bot_msg = {
            'id': str(uuid.uuid4()),
            'message': response,
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