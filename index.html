import os
import uuid
import sqlite3
import json
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai

# ── Config ──────────────────────────────────────────────
STRIPE_WEBHOOK   = os.getenv("STRIPE_WEBHOOK_SECRET", "")
ADMIN_SECRET     = os.getenv("ADMIN_SECRET", "changeme123")
DB_PATH          = os.getenv("DB_PATH", "licenses.db")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")

PRICE_MONTHLY    = os.getenv("STRIPE_PRICE_MONTHLY", "price_xxx")
PRICE_YEARLY     = os.getenv("STRIPE_PRICE_YEARLY", "price_xxx")
PRICE_LIFETIME   = os.getenv("STRIPE_PRICE_LIFETIME", "price_xxx")

PLANS = {
    PRICE_MONTHLY:  {"plan": "pro", "days": 31,    "label": "monthly"},
    PRICE_YEARLY:   {"plan": "pro", "days": 366,   "label": "yearly"},
    PRICE_LIFETIME: {"plan": "pro", "days": 99999, "label": "lifetime"},
}

# ── Gemini init ─────────────────────────────────────────
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")
else:
    gemini_model = None

# ── App ─────────────────────────────────────────────────
app = FastAPI(title="AI Shell License API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── DB ──────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS licenses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key         TEXT UNIQUE NOT NULL,
                plan        TEXT NOT NULL DEFAULT 'pro',
                email       TEXT,
                stripe_id   TEXT,
                created_at  TEXT NOT NULL,
                expires_at  TEXT,
                active      INTEGER NOT NULL DEFAULT 1,
                label       TEXT
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_key ON licenses(key)")
        db.commit()

init_db()

# ── Helpers ─────────────────────────────────────────────
def generate_key(prefix="AISH") -> str:
    raw = str(uuid.uuid4()).replace("-", "").upper()
    parts = [raw[i:i+4] for i in range(0, 16, 4)]
    return f"{prefix}-{'-'.join(parts)}"

def create_license(plan, days, email="", stripe_id="", label="") -> str:
    key = generate_key()
    created = datetime.utcnow().isoformat()
    expires = (datetime.utcnow() + timedelta(days=days)).isoformat() if days < 9999 else "lifetime"
    with get_db() as db:
        db.execute(
            "INSERT INTO licenses (key, plan, email, stripe_id, created_at, expires_at, label) VALUES (?,?,?,?,?,?,?)",
            (key, plan, email, stripe_id, created, expires, label)
        )
        db.commit()
    return key

def is_valid_license(key: str) -> bool:
    with get_db() as db:
        row = db.execute("SELECT * FROM licenses WHERE key = ?", (key.strip().upper(),)).fetchone()
    if not row or not row["active"]:
        return False
    expires = row["expires_at"]
    if expires and expires != "lifetime":
        try:
            if datetime.fromisoformat(expires) < datetime.utcnow():
                return False
        except Exception:
            pass
    return True

# ── Models ──────────────────────────────────────────────
class ValidateRequest(BaseModel):
    key: str

class ManualKeyRequest(BaseModel):
    plan: str = "pro"
    days: int = 366
    email: str = ""
    label: str = ""
    admin_secret: str

class ChatRequest(BaseModel):
    key: str
    message: str
    history: list = []   # [{"role": "user"|"model", "parts": "..."}]

# ── Endpoints ────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "AI Shell License API online", "version": "2.0"}

@app.post("/api/validate-key")
def validate_key(req: ValidateRequest):
    key = req.key.strip().upper()
    if not key:
        return {"valid": False, "error": "Clave vacía"}
    with get_db() as db:
        row = db.execute("SELECT * FROM licenses WHERE key = ?", (key,)).fetchone()
    if not row:
        return {"valid": False, "error": "Clave no encontrada"}
    if not row["active"]:
        return {"valid": False, "error": "Clave desactivada"}
    expires = row["expires_at"]
    if expires and expires != "lifetime":
        try:
            if datetime.fromisoformat(expires) < datetime.utcnow():
                return {"valid": False, "error": "Clave expirada"}
        except Exception:
            pass
    return {
        "valid": True,
        "plan": row["plan"],
        "expires": expires,
        "label": row["label"] or "pro",
        "email": row["email"] or "",
    }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # 1. Validar licencia
    if not is_valid_license(req.key):
        raise HTTPException(status_code=403, detail="Licencia inválida o expirada")

    # 2. Verificar Gemini disponible
    if not gemini_model:
        raise HTTPException(status_code=503, detail="Gemini API key no configurada")

    # 3. Construir historial
    history = []
    for msg in req.history:
        role = msg.get("role", "user")
        parts = msg.get("parts", "")
        if role in ("user", "model") and parts:
            history.append({"role": role, "parts": parts})

    # 4. Llamar a Gemini
    try:
        chat_session = gemini_model.start_chat(history=history)
        response = chat_session.send_message(req.message)
        return {"reply": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

@app.post("/api/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    if STRIPE_WEBHOOK:
        try:
            import stripe
            event = stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        try:
            event = json.loads(payload)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")
    if event.get("type") == "checkout.session.completed":
        session = event["data"]["object"]
        email = session.get("customer_details", {}).get("email", "")
        stripe_id = session.get("id", "")
        price_id = session.get("metadata", {}).get("price_id", "")
        plan_info = PLANS.get(price_id, {"plan": "pro", "days": 366, "label": "pro"})
        key = create_license(
            plan=plan_info["plan"],
            days=plan_info["days"],
            email=email,
            stripe_id=stripe_id,
            label=plan_info["label"]
        )
        print(f"License created: {key} for {email}")
    elif event.get("type") == "customer.subscription.deleted":
        stripe_id = event["data"]["object"].get("id", "")
        if stripe_id:
            with get_db() as db:
                db.execute("UPDATE licenses SET active=0 WHERE stripe_id=?", (stripe_id,))
                db.commit()
    return {"received": True}

@app.post("/api/admin/create-key")
def admin_create_key(req: ManualKeyRequest):
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Admin secret inválido")
    key = create_license(plan=req.plan, days=req.days, email=req.email, label=req.label)
    return {"key": key, "plan": req.plan, "days": req.days}

@app.get("/api/admin/keys")
def admin_list_keys(secret: str = ""):
    if secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Admin secret inválido")
    with get_db() as db:
        rows = db.execute("SELECT * FROM licenses ORDER BY id DESC LIMIT 100").fetchall()
    return [dict(r) for r in rows]

@app.post("/api/admin/revoke")
def admin_revoke(key: str, secret: str = ""):
    if secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Admin secret inválido")
    with get_db() as db:
        db.execute("UPDATE licenses SET active=0 WHERE key=?", (key.upper(),))
        db.commit()
    return {"revoked": key}
