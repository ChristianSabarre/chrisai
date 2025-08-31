import google.generativeai as genai
import chromadb
import json
import os
from typing import List, Dict, Any

API_KEY = "AIzaSyDBKFb-gfTHYgX9CR_55VtJnO6Vm3xTjXc" 
COLLECTION_NAME = "valorant_patches"
PATCH_NOTES_FILE = "feedme_patchnotes.json"

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
                print(f"Error generating embedding for text: {e}")
               
                embeddings.append([0.0] * 768)  
        return embeddings

    def name(self) -> str:
        return "gemini-embedding"

def setup_chromadb():
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=GeminiEmbedding()
        )
        return client, collection
    except Exception as e:
        print(f"Error setting up ChromaDB: {e}")
        return None, None

def load_patch_notes(filename: str) -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found!")
            return []
        
        with open(filename, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        print(f"Loaded {len(dataset)} patch notes from {filename}")
        return dataset
    except Exception as e:
        print(f"Error loading patch notes: {e}")
        return []

def add_documents_to_collection(collection, dataset: List[Dict[str, Any]]):
    if not dataset:
        print("No data to add to collection")
        return False
    
    try:
        documents = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(dataset):
            content = doc.get("final_content") 
            
            documents.append(content)
            metadatas.append({
                "source": "valorant_patch_notes",
                "patch_id": doc.get("patch_id", f"patch_{i}"),
                # "title": doc.get("title", f"Patch {i}")
            })
            ids.append(str(i))
        
        # Add documents in batches to avoid potential memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
        
        print(f"Successfully added {len(dataset)} patch notes to ChromaDB collection!")
        return True
    except Exception as e:
        print(f"Error adding documents to collection: {e}")
        return False

# --- RAG Chat Function ---
def chat_chris(prompt: str, collection, k: int = 3) -> str:
    try:
        # Query the collection for relevant documents
        results = collection.query(
            query_texts=[prompt],
            n_results=k
        )
        
        # Handle empty results
        if not results['documents'] or not results['documents'][0]:
            context = "No relevant patch notes found for your query."
        else:
            retrieved_docs = results['documents'][0]
            context = "\n\n".join(retrieved_docs)
        
        
        full_prompt = f"""You are Chris AI, an expert on Valorant patch notes. Answer the user's question based on the provided patch notes context.

Context from Valorant Patch Notes:
{context}

User Question: {prompt}

Respond like you're just talking to a friend about the game:

Examples of natural responses:
- "Oh yeah, they hit Jett pretty hard in 9.11 - increased her dash cooldown to 12 seconds, so you can't escape as easily anymore"
- "Haven't seen any Killjoy changes lately, but back in 6.08 they made her Nanoswarm easier to see and gave her turret less health"
- "Dude, 10.04 was wild - they added that new agent Waylay and rotated the maps again"

Keep it natural:
- Use casual language and contractions (don't, can't, they're)
- Add your own commentary on changes ("this was huge", "kinda needed", "players hated this")
- Explain what changes actually mean for gameplay
- Connect related changes ("since they nerfed X, Y became more popular")
- If info seems mixed up or irrelevant, focus on what actually answers their question
- Don't use formal bullet points - just explain things conversationally

If you don't have good info, just say something like "Hmm, I don't see any recent Killjoy changes in what I've got. Want to ask about a specific patch or another agent?
"""
        
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(full_prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    """Main function to run the RAG system"""
    print("Initializing Valorant Patch Notes RAG System...")
    
    client, collection = setup_chromadb()
    if not collection:
        print("Failed to setup ChromaDB. Exiting.")
        return
    
    dataset = load_patch_notes(PATCH_NOTES_FILE)
    if not dataset:
        print("No patch notes loaded. Exiting.")
        return
    
    if collection.count() == 0:
        print("Collection is empty. Adding patch notes...")
        if not add_documents_to_collection(collection, dataset):
            print("Failed to add documents. Exiting.")
            return
    else:
        print(f"Collection already contains {collection.count()} documents")
    
    print("\n" + "="*50)
    print("Chris AI - Valorant Patch Notes Assistant")
    print("Type 'exit' or 'quit' to end the conversation")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["exit", "quit", ""]:
                print("Goodbye!")
                break
            
            print("Chris AI: ", end="", flush=True)
            response = chat_chris(user_input, collection)
            print(response)
            print()  # Add spacing between responses
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()