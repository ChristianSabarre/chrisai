from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import chromadb
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')  # Use environment variable

# Configuration
API_KEY = os.environ.get('GEMINI_API_KEY', "AIzaSyDBKFb-gfTHYgX9CR_55VtJnO6Vm3xTjXc")  # Use environment variable
COLLECTION_NAME = "valorant_patches"
PATCH_NOTES_FILE = "feedme_patchnotes.json"
EMBEDDING_DIMENSION = 768

genai.configure(api_key=API_KEY)

class GeminiEmbedding:
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",  
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                logger.error(f"Error generating embedding for text: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * EMBEDDING_DIMENSION)  
        return embeddings

    def name(self) -> str:
        return "gemini-embedding"

def setup_chromadb():
    """Initialize ChromaDB client and collection."""
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=GeminiEmbedding()
        )
        logger.info("ChromaDB setup successful")
        return client, collection
    except Exception as e:
        logger.error(f"Error setting up ChromaDB: {e}")
        return None, None

# Global variables for patch metadata
dataset = []
latest_patch = None
oldest_patch = None
client = None
collection = None

def load_patch_notes(filename: str) -> List[Dict[str, Any]]:
    """Load and sort patch notes from JSON file."""
    global dataset, latest_patch, oldest_patch
    
    try:
        if not os.path.exists(filename):
            logger.error(f"File '{filename}' not found!")
            return []

        with open(filename, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # Sort by published date
        dataset_sorted = sorted(
            dataset,
            key=lambda x: datetime.strptime(x.get("published", "1970-01-01"), "%Y-%m-%d")
        )

        # Identify oldest and latest patches
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
    """Add patch note documents to ChromaDB collection."""
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
            
            if content.strip():  # Only add if there's actual content
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
        
        # Add documents in batches
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
    
    if len(message) > 1000:  # Reasonable limit
        return "Message too long (max 1000 characters)"
    
    # Basic sanitization - remove potentially harmful characters
    dangerous_chars = ['<script', '</script', 'javascript:', 'data:']
    message_lower = message.lower()
    for char in dangerous_chars:
        if char in message_lower:
            return "Message contains invalid characters"
    
    return None

def chat_chris(prompt: str, collection, k: int = 5) -> str:
    """Generate AI response using RAG with ChromaDB and Gemini."""
    try:
        # Validate input
        validation_error = validate_message(prompt)
        if validation_error:
            return f"Error: {validation_error}"
        
        # Query the collection for relevant documents
        results = collection.query(
            query_texts=[prompt],
            n_results=k
        )
        
        # Handle empty results
        if not results['documents'] or not results['documents'][0]:
            context = "No relevant patch notes found for your query."
            retrieved_docs = []
            retrieved_metadata = []
        else:
            retrieved_docs = results['documents'][0]
            retrieved_metadata = results['metadatas'][0] if results['metadatas'] else []

        # Get current date and patch info
        current_date = datetime.now().strftime("%Y-%m-%d")
        latest_title = latest_patch.get("title", "N/A") if latest_patch else "N/A"
        latest_date = latest_patch.get("published", "N/A") if latest_patch else "N/A"
        oldest_title = oldest_patch.get("title", "N/A") if oldest_patch else "N/A"
        oldest_date = oldest_patch.get("published", "N/A") if oldest_patch else "N/A"

        # Create enhanced context with metadata
        enhanced_context_parts = []
        for i, doc in enumerate(retrieved_docs):
            metadata = retrieved_metadata[i] if i < len(retrieved_metadata) else {}
            patch_title = metadata.get('title', 'Unknown Patch')
            patch_date = metadata.get('published', 'Unknown Date')
            
            enhanced_context_parts.append(f"[PATCH: {patch_title} - Published: {patch_date}]\n{doc}")

        # Automatically include latest patch content if not already in results
        if latest_patch:
            latest_content = latest_patch.get("final_content", "")
            if latest_content and latest_content not in retrieved_docs:
                enhanced_context_parts.insert(0, f"[LATEST PATCH: {latest_title} - Published: {latest_date}]\n{latest_content}")

        # Combine context for the model
        context = "\n\n---\n\n".join(enhanced_context_parts) if enhanced_context_parts else "No relevant patch notes found."

        # Create chronological awareness
        def calculate_recency(date_str):
            try:
                patch_date = datetime.strptime(date_str, "%Y-%m-%d")
                current = datetime.strptime(current_date, "%Y-%m-%d")
                days_ago = (current - patch_date).days
                
                if days_ago < 30:
                    return "very recent"
                elif days_ago < 90:
                    return "recent"
                elif days_ago < 365:
                    return "from this year"
                elif days_ago < 730:
                    return "from last year"
                else:
                    return "older"
            except:
                return "unknown timing"

        latest_recency = calculate_recency(latest_date) if latest_date != "N/A" else "unknown"

        full_prompt = f"""You are Chris AI, an expert on Valorant patch notes. Answer the user's question based on the provided patch notes context.

IMPORTANT TEMPORAL CONTEXT:
- Current date: {current_date}
- Latest patch: {latest_title} (published {latest_date}) - This is {latest_recency}
- Oldest patch: {oldest_title} (published {oldest_date})
- Your data spans from {oldest_date} to {latest_date}

Context from Valorant Patch Notes:
{context}

User Question: {prompt}

CRITICAL INSTRUCTIONS FOR TEMPORAL AWARENESS:
- When users ask about "recent" changes, they mean patches from 2024-2025
- When users ask about "latest" or "newest", focus on patches from late 2024 or 2025
- When users ask about "current meta" or "now", reference the most recent patches available
- Always check the publication dates in the context to understand when changes happened
- If a patch is from 2022-2023, mention it's from a while back: "Back in 2023 they changed X, but more recently in 2024..."
- Don't treat 2022-2023 patches as current unless specifically asked about historical changes

Respond like you're just talking to a friend about the game:

Examples of temporally-aware responses:
- "Yeah, in the latest patch from December 2024 they nerfed Jett's dash cooldown"
- "That was changed way back in 2022, but the current version since patch 8.11 in 2024 works differently"
- "Haven't seen any Killjoy changes in the recent patches - last major change was back in early 2023"

Keep it natural:
- Use casual language and contractions (don't, can't, they're)
- Add temporal context ("recently", "back in", "currently", "as of the latest patch")
- Explain what changes actually mean for current gameplay
- If mentioning old changes, clarify they're not recent: "that was back in 2022 though"
- Prioritize newer information when multiple time periods are relevant

If you don't have good recent info, say something like "Hmm, I don't see any recent changes for that. The last time they touched it was back in [year]. Want to ask about something else or a specific patch?"
"""
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(full_prompt)
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I'm having trouble processing your request right now. Try asking again in a moment!"

def initialize_system():
    """Initialize the system components."""
    global client, collection
    
    logger.info("Initializing Valorant Patch Notes RAG System...")
    
    # Setup ChromaDB
    client, collection = setup_chromadb()
    if not collection:
        logger.error("Failed to setup ChromaDB.")
        return False
    
    # Load patch notes
    dataset_sorted = load_patch_notes(PATCH_NOTES_FILE)
    if not dataset_sorted:
        logger.error("No patch notes loaded.")
        return False
    
    # Add documents to collection if empty
    if collection.count() == 0:
        logger.info("Collection is empty. Adding patch notes...")
        if not add_documents_to_collection(collection, dataset_sorted):
            logger.error("Failed to add documents.")
            return False
    else:
        logger.info(f"Collection already contains {collection.count()} documents")
    
    logger.info("System initialization complete!")
    return True

# Initialize system when module is imported
initialize_system()

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        # Validate message
        validation_error = validate_message(user_message)
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        # Check system status
        if not collection:
            return jsonify({'error': 'System not initialized properly'}), 500
        
        # Generate response
        response = chat_chris(user_message, collection)
        
        # Create chat message objects
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
    """Health check endpoint."""
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
    """Get system statistics."""
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