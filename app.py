import chromadb
import json
import os
from typing import List, Dict, Any

from mistral_client import MistralEmbedding, chat as mistral_chat
from prompts import build_system_prompt, build_user_prompt, describe_corpus

COLLECTION_NAME = "valorant_patches"
PATCH_NOTES_FILE = "feedme_patchnotes.json"

def setup_chromadb():
    try:
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=MistralEmbedding()
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
def chat_chris(prompt: str, collection, dataset: List[Dict[str, Any]], k: int = 3) -> str:
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
        
        
        corpus = describe_corpus(dataset)
        system_prompt = build_system_prompt(corpus)
        user_prompt = build_user_prompt(context, prompt)

        return mistral_chat(system_prompt, user_prompt)

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
            response = chat_chris(user_input, collection, dataset)
            print(response)
            print()  # Add spacing between responses
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()