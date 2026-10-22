# proxy.py — AI Proxy Server for Groq
# One file, deployed on multiple Render accounts with different env

import os
import httpx
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# ============ CONFIG FROM ENV ============
PROXY_ID      = os.getenv("PROXY_ID", "proxy_unknown")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
PROXY_SECRET  = os.getenv("PROXY_SECRET", "")

# ============ MODELS (HARDCODED) ============
# এগুলো env-এ রাখার দরকার নেই — ৬ সার্ভারে একই
GROQ_TEXT_MODEL   = "openai/gpt-oss-120b"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============ HTTP CLIENT ============
http_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    print(f"[{PROXY_ID}] started | text={GROQ_TEXT_MODEL} | vision={GROQ_VISION_MODEL} | groq_key={'SET' if GROQ_API_KEY else 'MISSING'}")
    yield
    await http_client.aclose()
    print(f"[{PROXY_ID}] stopped")

app = FastAPI(title=f"AI Proxy {PROXY_ID}", lifespan=lifespan)


def has_image_content(messages) -> bool:
    """OpenAI-style messages এ image_url আছে কিনা"""
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


# ============ MAIN ENDPOINT ============
@app.post("/chat/completions")
async def proxy_chat(request: Request, x_proxy_key: str = Header(None)):
    """Forward Groq-compatible chat request to Groq API"""
    # 1. Auth
    if not PROXY_SECRET or x_proxy_key != PROXY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid proxy key")

    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Proxy not configured")

    # 2. Parse body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 3. 🔀 Text vs Vision auto-route
    if has_image_content(body.get("messages")):
        body["model"] = GROQ_VISION_MODEL
        print(f"[{PROXY_ID}] → VISION: {GROQ_VISION_MODEL}")
    else:
        body["model"] = GROQ_TEXT_MODEL
        print(f"[{PROXY_ID}] → TEXT: {GROQ_TEXT_MODEL}")

    # 4. Forward to Groq
    try:
        response = await http_client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=body
        )

        if response.status_code >= 400:
            return JSONResponse(
                status_code=response.status_code,
                content=response.json()
            )

        return response.json()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"[{PROXY_ID}] Groq timeout")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"[{PROXY_ID}] Proxy error: {str(e)}")


# ============ HEALTH & INFO ============
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "new",
        "proxy_id": PROXY_ID,
        "text_model": GROQ_TEXT_MODEL,
        "vision_model": GROQ_VISION_MODEL,
        "groq_configured": bool(GROQ_API_KEY),
        "secret_configured": bool(PROXY_SECRET),
    }

@app.get("/")
async def root():
    return {"service": "AI Proxy", "proxy_id": PROXY_ID, "status": "running"}

@app.get("/my-ip")
async def get_my_ip():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("https://api.ipify.org?format=json")
            return {"proxy_id": PROXY_ID, **r.json()}
    except Exception as e:
        return {"proxy_id": PROXY_ID, "error": str(e)}


# ============ RUN ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000"))
    )