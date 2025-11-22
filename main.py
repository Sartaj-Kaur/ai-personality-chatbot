from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import uvicorn
from dotenv import load_dotenv
import os

# ---------------------------------------
# 1. CONFIGURE GROQ API
# ---------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = "llama-3.1-8b-instant"

# ---------------------------------------
# 2. FASTAPI + CORS
# ---------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# 3. REQUEST BODY
# ---------------------------------------
class ChatRequest(BaseModel):
    message: str
    personality: str = "friendly"

# Optional: define personality prompts
PERSONALITY_PROMPTS = {
    "friendly": "You are a friendly AI assistant. Be polite, kind, and helpful.",
    "humorous": "You are a humorous AI assistant. Add light jokes or fun comments when appropriate.",
    "professional": "You are a professional AI assistant. Be concise, clear, and formal.",
    "helpful": "You are a helpful AI assistant. Focus on giving clear, step-by-step guidance.",
    "bestie": "You are a fun and girly bestie AI. Be cheerful, encouraging, and use casual friendly language."
}

# ---------------------------------------
# 4. CHAT ENDPOINT
# ---------------------------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    # Pick the personality prompt
    personality_prompt = PERSONALITY_PROMPTS.get(req.personality, PERSONALITY_PROMPTS["friendly"])

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": personality_prompt},
                {"role": "user", "content": req.message}
            ]
        )

        ai_response = completion.choices[0].message.content
        return {"response": ai_response}

    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------
# 5. RUN SERVER
# ---------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
