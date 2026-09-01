import os
import json
import asyncio
import time
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
# Use the newer router endpoint (more reliable)
HF_API_URLS = [
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}",
    f"https://api-inference.huggingface.co/models/{HF_MODEL}"  # fallback old endpoint
]

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

SYSTEM_PROMPT = """You are Noussa, a personal AI assistant like Jarvis from Iron Man.
You have access to the user's device context (screen, files, etc.) and long-term memory.
Be concise, loyal, and helpful. Address the user as 'sir' or 'ma'am'.
Current context: {context}
Relevant memories: {memories}
"""

@app.get("/health")
async def health_check():
    return {"status": "alive"}

def query_huggingface(prompt_text):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt_text,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    # Try each endpoint with retries
    for url in HF_API_URLS:
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]["generated_text"].strip()
                    elif isinstance(result, dict) and "generated_text" in result:
                        return result["generated_text"].strip()
                    else:
                        raise Exception(f"Unexpected response format: {result}")
                else:
                    # If 4xx or 5xx, maybe endpoint is wrong; try next endpoint
                    print(f"Attempt {attempt+1}: HTTP {response.status_code} from {url}, response: {response.text[:200]}")
                    if response.status_code == 404:
                        break  # model not found, no point retrying
                    time.sleep(2 ** attempt)  # exponential backoff
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: Network error from {url}: {e}")
                time.sleep(2 ** attempt)
    raise Exception("All Hugging Face endpoints failed after retries")

def retrieve_memories(query, limit=5):
    try:
        response = supabase.table("memories").select("text").limit(50).execute()
        all_memories = [row["text"] for row in response.data]
        query_words = set(query.lower().split())
        scored = []
        for mem in all_memories:
            mem_words = set(mem.lower().split())
            score = len(query_words.intersection(mem_words))
            scored.append((score, mem))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [mem for _, mem in scored[:limit]]
    except Exception as e:
        print(f"Memory retrieval error: {e}")
        return []

def add_memory(text, importance=5):
    try:
        supabase.table("memories").insert({
            "text": text,
            "importance": importance,
            "embedding": [0.0] * 1536
        }).execute()
    except Exception as e:
        print(f"Memory insert error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if "ping" in msg:
                await websocket.send_text(json.dumps({"response": "pong"}))
                continue

            user_input = msg.get("text", "")
            device_context = msg.get("context", {})
            context_str = f"Device: {device_context.get('device', 'unknown')}\nScreen: {device_context.get('screen', 'none')}"

            memories = retrieve_memories(user_input)
            memories_str = "\n".join(memories) if memories else "No relevant memories."

            prompt = SYSTEM_PROMPT.format(context=context_str, memories=memories_str)
            full_prompt = f"{prompt}\n\nUser: {user_input}\nAssistant:"

            try:
                reply = query_huggingface(full_prompt)
            except Exception as e:
                reply = f"I'm sorry, sir. I encountered an error while processing your request: {str(e)}"
                print(f"HF error: {e}")

            add_memory(f"User: {user_input}\nAssistant: {reply}")

            await websocket.send_text(json.dumps({"response": reply}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket outer error: {e}")
