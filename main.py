import os
import json
import time
import sys
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

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

def query_openrouter(prompt_text):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }
    for attempt in range(3):
        try:
            response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
            print(f"OpenRouter attempt {attempt+1}: status {response.status_code}", flush=True)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"OpenRouter error body: {response.text[:300]}", flush=True)
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"OpenRouter network error: {e}", flush=True)
            time.sleep(2 ** attempt)
    raise Exception("OpenRouter API failed after retries")

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
        print(f"Memory retrieval error: {e}", flush=True)
        return []

def add_memory(text, importance=5):
    try:
        supabase.table("memories").insert({
            "text": text,
            "importance": importance,
            "embedding": [0.0] * 1536
        }).execute()
    except Exception as e:
        print(f"Memory insert error: {e}", flush=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected", flush=True)
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
                reply = query_openrouter(full_prompt)
            except Exception as e:
                reply = f"Error: {str(e)}"
                print(f"OpenRouter error: {e}", flush=True)

            add_memory(f"User: {user_input}\nAssistant: {reply}")

            await websocket.send_text(json.dumps({"response": reply}))

    except WebSocketDisconnect:
        print("Client disconnected", flush=True)
    except Exception as e:
        print(f"WebSocket outer error: {e}", flush=True)
