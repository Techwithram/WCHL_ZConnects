# app.py — Z Connects (robust & offline-friendly)
import os
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer, util
import numpy as np
from fastapi import FastAPI, Form, Request, HTTPException, Response,Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

# ------------------ CONFIG ------------------
DATA_FILE = "data_store.json"
MODEL_NAME = "all-MiniLM-L6-v2"          # tries this first; falls back to local embedder
COMMENT_WINDOW_DAYS = 14
MATCH_THRESHOLD = 0.35                    # /chats shows >= this; change if you like

WEIGHTS = {                               # exact split you asked for
    "present_comment": 0.50,
    "profile": 0.38,
    "recent_activity": 0.12,
}

ABUSE_WORDS = {"abuse", "stupid", "idiot", "hate", "kill"}

# Ensure folders exist (avoids common runtime errors)
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ------------------ APP ---------------------
app = FastAPI(title="Z Connects — Dynamic Prototype")
app.mount("/templates", StaticFiles(directory="templates"), name="templates")
templates = Jinja2Templates(directory="templates")
@app.get("/hello")
def hello():
    print("✅ /hello hit")
    return {"msg": "Hello works!"}


# ------------------ Embedding backend ------------------
class LocalHashEmbedder:
    """
    Offline-safe bag-of-words hashing embedder.
    384-d vectors by hashing tokens into bins; L2 normalized.
    Deterministic and fast; not SOTA but reliable with no internet.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec
        for tok in text.lower().split():
            h = hash(tok) % self.dim
            vec[h] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        if isinstance(texts, str):
            return self._embed_text(texts)
        return np.vstack([self._embed_text(t) for t in texts])

# Try to load SentenceTransformer; fall back if needed
_embedder = SentenceTransformer(MODEL_NAME)
try:
    from sentence_transformers import SentenceTransformer
    _embedder = SentenceTransformer(MODEL_NAME)
    print("[INFO] Loaded SentenceTransformer:", MODEL_NAME)
except Exception as e:
    print("[WARN] Could not load SentenceTransformer:", e)
    print("[INFO] Falling back to LocalHashEmbedder (offline mode).")
    _embedder = LocalHashEmbedder(dim=384)

def embed_text_safe(text: str) -> np.ndarray:
    """
    Uses _embedder.encode if available; always returns a 1D numpy float32 vector (normalized).
    """
    try:
        emb = _embedder.encode(text, convert_to_numpy=True)
    except Exception as e:
        # if embedder fails, return zero vector sized 384 (LocalHashEmbedder fallback should prevent this)
        print("[WARN] embed_text_safe encode failed:", e)
        return np.zeros(384, dtype=np.float32)

    if emb is None:
        return np.zeros(384, dtype=np.float32)

    emb = np.array(emb, dtype=np.float32)
    if emb.ndim > 1:
        emb = emb[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb

# ------------------ DATA --------------------
db: Dict[str, Any] = {"users": [], "posts": [], "comments": [], "matches": [], "messages": []}
emb_cache: Dict[str, Dict[str, np.ndarray]] = {"user_profiles": {}, "comments": {}}
sessions: Dict[str, str] = {}  # session_token -> user_id

# ------------------ PERSISTENCE -------------
def save_data():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_data():
    global db
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                db = json.load(f)
        except Exception:
            db = {"users": [], "posts": [], "comments": [], "matches": [], "messages": []}
    else:
        save_data()

# ------------------ UTILS -------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def parse_iso(s: str) -> datetime:
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    # Make sure it's always timezone-aware (UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def cosine_sim_safe(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    try:
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=np.float32)
        if not isinstance(b, np.ndarray):
            b = np.array(b, dtype=np.float32)
        if a.size == 0 or b.size == 0:
            return 0.0
        an = np.linalg.norm(a); bn = np.linalg.norm(b)
        if an == 0.0 or bn == 0.0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))
    except Exception as e:
        print("[WARN] cosine_sim_safe error:", e)
        return 0.0

def is_abusive(text: str) -> bool:
    words = set(text.lower().split())
    return any(bad in words for bad in ABUSE_WORDS)

# ------------------ EMBEDDING CACHE HELPERS --------------
def cache_user_embedding(user: dict):
    """
    Ensure emb_cache['user_profiles'][user_id] has a numpy array and user['embedding'] contains a list.
    """
    uid = user.get("user_id")
    if not uid:
        return
    arr = None
    # if embedding present in JSON, attempt to use it
    if user.get("embedding"):
        try:
            arr = np.array(user["embedding"], dtype=np.float32)
        except Exception:
            arr = None
    if arr is None or arr.size == 0:
        arr = embed_text_safe(user.get("profile_text", "") or "")
        # persist back to JSON-friendly list
        user["embedding"] = arr.tolist()
    emb_cache["user_profiles"][uid] = arr

def cache_comment_embedding(comment: dict):
    """
    Ensure emb_cache['comments'][comment_id] has numpy array and comment['embedding'] has list.
    """
    cid = comment.get("comment_id")
    if not cid:
        return
    arr = None
    if comment.get("embedding"):
        try:
            arr = np.array(comment["embedding"], dtype=np.float32)
        except Exception:
            arr = None
    if arr is None or arr.size == 0:
        arr = embed_text_safe(comment.get("text", "") or "")
        comment["embedding"] = arr.tolist()
    emb_cache["comments"][cid] = arr

def get_user_emb(user_id: str) -> Optional[np.ndarray]:
    """Return numpy embedding for user_id, hydrating from db if needed."""
    emb = emb_cache["user_profiles"].get(user_id)
    if emb is not None:
        return emb
    # hydrate from db
    u = next((x for x in db["users"] if x["user_id"] == user_id), None)
    if not u:
        return None
    cache_user_embedding(u)
    return emb_cache["user_profiles"].get(user_id)

def get_user_connections(user_id: str):
    """
    Returns a list of dicts: [{user_id, name, score}], unique per other user.
    Keeps the highest score across interactions.
    """
    best = {}
    for m in db.get("matches", []):
        if m.get("score", 0) < MATCH_THRESHOLD:
            continue
        if m["user_a_id"] == user_id:
            other_id = m["user_b_id"]; other_name = m["user_b_name"]
        elif m["user_b_id"] == user_id:
            other_id = m["user_a_id"]; other_name = m["user_a_name"]
        else:
            continue
        if (other_id not in best) or (m["score"] > best[other_id]["score"]):
            best[other_id] = {"user_id": other_id, "name": other_name, "score": m["score"]}
    return list(best.values())


def get_comment_emb(comment_id: str) -> Optional[np.ndarray]:
    """Return numpy embedding for comment_id, hydrating from db if needed."""
    emb = emb_cache["comments"].get(comment_id)
    if emb is not None:
        return emb
    c = next((x for x in db["comments"] if x["comment_id"] == comment_id), None)
    if not c:
        return None
    cache_comment_embedding(c)
    return emb_cache["comments"].get(comment_id)

def rebuild_all_matches():
    print("[INFO] Rebuilding all matches with fresh embeddings...")
    db["matches"].clear()
    pairs_seen = {}

    for i, c1 in enumerate(db.get("comments", [])):
        for j, c2 in enumerate(db.get("comments", [])):
            if j <= i: 
                continue
            if c1["post_id"] != c2["post_id"]:
                continue

            u1 = next((u for u in db["users"] if u["user_id"] == c1["user_id"]), None)
            u2 = next((u for u in db["users"] if u["user_id"] == c2["user_id"]), None)
            if not u1 or not u2 or u1["user_id"] == u2["user_id"]:
                continue

            score, details = compute_match_score(u1, u2, c1, c2)
            if score < MATCH_THRESHOLD:
                continue

        # canonical key (user_a,user_b)
            key = tuple(sorted([u1["user_id"], u2["user_id"]]))
            if key not in pairs_seen or score > pairs_seen[key]["score"]:
                pairs_seen[key] = {
                    "user_a_id": key[0],
                    "user_b_id": key[1],
                    "user_a_name": u1["display_name"] if u1["user_id"] == key[0] else u2["display_name"],
                    "user_b_name": u2["display_name"] if u2["user_id"] == key[1] else u1["display_name"],
                    "score": score,
                    "details": details,
                    "post_id": c1["post_id"],
                    "timestamp": datetime.utcnow().isoformat()
                }

    db["matches"].extend(pairs_seen.values())
    save_data() 
# ------------------ AUTH HELPERS -------------------------
def get_current_user(request: Request) -> str:
    token = request.cookies.get("session_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Not logged in")
    return sessions[token]

# ------------------ MATCHING -----------------------------
def recent_activity_score(target_user_id: str, ref_comment_emb: Optional[np.ndarray], now_dt: datetime) -> float:
    """
    Similarity of the new comment to the target user's own recent comments (last 14 days).
    Returns average of top-3 similarities (or 0 if none).
    """
    if ref_comment_emb is None:
        return 0.0
    start = now_dt - timedelta(days=COMMENT_WINDOW_DAYS)
    sims: List[float] = []
    for c in db["comments"]:
        if c["user_id"] != target_user_id:
            continue
        try:
            ts = parse_iso(c["timestamp"])
        except Exception:
            continue
        if ts < start:
            continue
        emb = get_comment_emb(c["comment_id"])
        sims.append(cosine_sim_safe(ref_comment_emb, emb))
    if not sims:
        return 0.0
    sims.sort(reverse=True)
    top = sims[:3]
    return float(sum(top) / len(top))

def compute_match_score(user_a: dict, user_b: dict, comment_a: dict, comment_b: dict):
    """
    Returns: (score: float, details: dict)
    Uses get_user_emb / get_comment_emb which auto-hydrate the cache.
    """
    # fetch embeddings safely (auto-hydrate if needed)
    c_emb_a = get_comment_emb(comment_a["comment_id"])
    c_emb_b = get_comment_emb(comment_b["comment_id"])
    p_emb_a = get_user_emb(user_a["user_id"])
    p_emb_b = get_user_emb(user_b["user_id"])

    # debug prints (will show whether embeddings were found)
    print(f"[DEBUG] compute_match_score: c_emb_a found? {c_emb_a is not None}, c_emb_b found? {c_emb_b is not None}")
    print(f"[DEBUG]                 p_emb_a found? {p_emb_a is not None}, p_emb_b found? {p_emb_b is not None}")

    comment_sim = cosine_sim_safe(c_emb_a, c_emb_b)
    profile_sim = cosine_sim_safe(p_emb_a, p_emb_b)

    # recent activity similarity: average-top-3 of user_b's recent comments vs new comment
    recent_sim = 0.0
    try:
        recent_sim = recent_activity_score(user_b["user_id"], c_emb_a, datetime.utcnow())
    except Exception as e:
        print("[WARN] recent_activity_score failed:", e)
        recent_sim = 0.0

    # compute final score with your weights
    score = WEIGHTS["present_comment"] * comment_sim + WEIGHTS["profile"] * profile_sim + WEIGHTS["recent_activity"] * recent_sim

    details = {
        "comment_sim": float(comment_sim),
        "profile_sim": float(profile_sim),
        "recent_activity_sim": float(recent_sim)
    }
    print(f"[DEBUG] score breakdown: comment_sim={comment_sim}, profile_sim={profile_sim}, recent_sim={recent_sim}, score={score}")
    return float(score), details

# def auto_match_for_comment(new_comment: dict):
#     now_dt = datetime.utcnow()
#     post_id = new_comment["post_id"]
#     new_time = parse_iso(new_comment["timestamp"])
#     user_a = next((u for u in db["users"] if u["user_id"] == new_comment["user_id"]), None)
#     if not user_a:
#         return []

#     created = []
#     for c in db["comments"]:
#         if c["comment_id"] == new_comment["comment_id"]:
#             continue
#         # require same post (your design) - if you want cross-post matching, remove this check
#         if c["post_id"] != post_id:
#             continue
#         # time window
#         c_time = parse_iso(c["timestamp"])
#         if abs((new_time - c_time).days) > COMMENT_WINDOW_DAYS:
#             continue

#         user_b = next((u for u in db["users"] if u["user_id"] == c["user_id"]), None)
#         if not user_b:
#             continue
#         # skip if same user
#         if user_b["user_id"] == user_a["user_id"]:
#             continue

#         score, details = compute_match_score(user_a, user_b, new_comment, c)

#         if score >= MATCH_THRESHOLD:
#             # avoid duplicate pair entries (a-b vs b-a)
#             existing = any(
#                 (m.get("user_a_id") == user_a["user_id"] and m.get("user_b_id") == user_b["user_id"]) or
#                 (m.get("user_a_id") == user_b["user_id"] and m.get("user_b_id") == user_a["user_id"])
#                 for m in db["matches"]
#             )
#             if existing:
#                 print("[DEBUG] match already exists for pair:", user_a["user_id"], user_b["user_id"])
#                 continue

#             match_obj = {
#                 "user_a_id": user_a["user_id"],
#                 "user_b_id": user_b["user_id"],
#                 "user_a_name": user_a.get("display_name", user_a["user_id"]),
#                 "user_b_name": user_b.get("display_name", user_b["user_id"]),
#                 "score": score,
#                 "details": details,
#                 "post_id": post_id,
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#             db["matches"].append(match_obj)
#             created.append(match_obj)

#     if created:
#         save_data()
#     print(f"[DEBUG] auto_match_for_comment created {len(created)} matches")
#     return created

# def auto_match_for_comment(new_comment):
#     post_id = new_comment["post_id"]
#     new_time = datetime.fromisoformat(new_comment["timestamp"])
#     matches = []

#     for c in db["comments"]:
#         if c["comment_id"] == new_comment["comment_id"]:
#             continue
#         if c["post_id"] != post_id:
#             continue
#         c_time = datetime.fromisoformat(c["timestamp"])
#         if abs((new_time - c_time).days) > COMMENT_WINDOW_DAYS:
#             continue

#         user_a = next(u for u in db["users"] if u["user_id"] == new_comment["user_id"])
#         user_b = next(u for u in db["users"] if u["user_id"] == c["user_id"])

#         score, details = compute_match_score(user_a, user_b, new_comment, c)

#         if score >= MATCH_THRESHOLD:
#             # Build a unique identifier for the match
#             match_key = {
#                 "user_a_id": user_a["user_id"],
#                 "user_b_id": user_b["user_id"],
#                 "post_id": post_id
#             }

#             # Check if this match already exists
#             already_exists = any(
#                 m["user_a_id"] == match_key["user_a_id"] and
#                 m["user_b_id"] == match_key["user_b_id"] and
#                 m["post_id"] == match_key["post_id"]
#                 for m in db["matches"]
#             )

#             if not already_exists:
#                 match_obj = {
#                     "user_a_id": user_a["user_id"],
#                     "user_b_id": user_b["user_id"],
#                     "user_a_name": user_a["display_name"],
#                     "user_b_name": user_b["display_name"],
#                     "score": score,
#                     "details": details,
#                     "post_id": post_id,
#                     "timestamp": datetime.utcnow().isoformat()
#                 }
#                 db["matches"].append(match_obj)
#                 matches.append(match_obj)

#     save_data()
#     return matches

# def auto_match_for_comment(new_comment):
#     post_id = new_comment["post_id"]
#     new_time = datetime.fromisoformat(new_comment["timestamp"])
#     matches = []

#     for c in db["comments"]:
#         if c["comment_id"] == new_comment["comment_id"]:
#             continue
#         if c["post_id"] != post_id:
#             continue
#         c_time = datetime.fromisoformat(c["timestamp"])
#         if abs((new_time - c_time).days) > COMMENT_WINDOW_DAYS:
#             continue

#         user_a = next(u for u in db["users"] if u["user_id"] == new_comment["user_id"])
#         user_b = next(u for u in db["users"] if u["user_id"] == c["user_id"])

#         score, details = compute_match_score(user_a, user_b, new_comment, c)

#         if score >= MATCH_THRESHOLD:
#             # Build a unique identifier for the match
#             match_key = {
#                 "user_a_id": user_a["user_id"],
#                 "user_b_id": user_b["user_id"],
#                 "post_id": post_id
#             }

#             # Check if this match already exists
#             already_exists = any(
#                 m["user_a_id"] == match_key["user_a_id"] and
#                 m["user_b_id"] == match_key["user_b_id"] and
#                 m["post_id"] == match_key["post_id"]
#                 for m in db["matches"]
#             )

#             if not already_exists:
#                 match_obj = {
#                     "user_a_id": user_a["user_id"],
#                     "user_b_id": user_b["user_id"],
#                     "user_a_name": user_a["display_name"],
#                     "user_b_name": user_b["display_name"],
#                     "score": score,
#                     "details": details,
#                     "post_id": post_id,
#                     "timestamp": datetime.utcnow().isoformat()
#                 }
#                 db["matches"].append(match_obj)
#                 matches.append(match_obj)

#     save_data()
#     return matches

def auto_match_for_comment(new_comment):
    post_id = new_comment["post_id"]
    new_time = datetime.fromisoformat(new_comment["timestamp"])
    updated_matches = []

    for c in db["comments"]:
        if c["comment_id"] == new_comment["comment_id"]:
            continue
        if c["post_id"] != post_id:
            continue
        c_time = datetime.fromisoformat(c["timestamp"])
        if abs((new_time - c_time).days) > COMMENT_WINDOW_DAYS:
            continue

        user_a = next(u for u in db["users"] if u["user_id"] == new_comment["user_id"])
        user_b = next(u for u in db["users"] if u["user_id"] == c["user_id"])

        # Skip self-match
        if user_a["user_id"] == user_b["user_id"]:
            continue

        score, details = compute_match_score(user_a, user_b, new_comment, c)

        if score >= MATCH_THRESHOLD:
            # --- GLOBAL uniqueness check ---
            existing = next(
    (m for m in db["matches"]
     if ((m["user_a_id"] == user_a["user_id"] and m["user_b_id"] == user_b["user_id"]) or
         (m["user_a_id"] == user_b["user_id"] and m["user_b_id"] == user_a["user_id"]))),
    None
)

            if existing:
        # update if score is higher
                if score > existing["score"]:
                    existing["score"] = score
                    existing["details"] = details
                    existing["post_id"] = post_id
                    existing["timestamp"] = datetime.utcnow().isoformat()
            else:
                match_obj = {
                    "user_a_id": user_a["user_id"],
                    "user_b_id": user_b["user_id"],
                    "user_a_name": user_a["display_name"],
                    "user_b_name": user_b["display_name"],
                    "score": score,
                    "details": details,
                    "post_id": post_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
                db["matches"].append(match_obj)
                updated_matches.append(match_obj)
                print(f"[DEBUG] New global match {user_a['user_id']} & {user_b['user_id']} (score={score})")

    save_data()
    return updated_matches



# ------------------ SCHEMA NORMALIZER --------------------
def normalize_matches_schema():
    """
    If older matches used keys 'user_a'/'user_b', convert them to 'user_a_id'/'user_b_id'
    so /chats can render them.
    """
    changed = False
    for m in db.get("matches", []):
        if "user_a_id" not in m and "user_a" in m:
            m["user_a_id"] = m.pop("user_a")
            changed = True
        if "user_b_id" not in m and "user_b" in m:
            m["user_b_id"] = m.pop("user_b")
            changed = True
        # Fill names if missing
        if "user_a_name" not in m:
            ua = next((u for u in db["users"] if u["user_id"] == m.get("user_a_id")), None)
            if ua:
                m["user_a_name"] = ua.get("display_name", m.get("user_a_id"))
                changed = True
        if "user_b_name" not in m:
            ub = next((u for u in db["users"] if u["user_id"] == m.get("user_b_id")), None)
            if ub:
                m["user_b_name"] = ub.get("display_name", m.get("user_b_id"))
                changed = True
    if changed:
        save_data()

# ------------------ PAGES -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root_redirect():
    return RedirectResponse(url="/login_page", status_code=303)

@app.get("/login_page", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login1.html", {"request": request})

@app.get("/feed", response_class=HTMLResponse)
async def feed(request: Request):
    try:
        user_id = get_current_user(request)
    except HTTPException:
        return RedirectResponse(url="/login_page", status_code=303)

    posts = sorted(db["posts"], key=lambda p: p.get("created_at", ""), reverse=True)
    return templates.TemplateResponse("feed.html", {
        "request": request,
        "posts": posts,
        "comments": db["comments"],
        "me": user_id
    })
# Chats are disabled in this version; use /matches instead
# @app.get("/chats", response_class=HTMLResponse)
# async def chats_page(request: Request):
#     user_id = get_current_user(request)
#     # Filter only valid matches where score >= threshold
#     user_matches = [m for m in db["matches"] 
#                     if (m["user_a_id"] == user_id or m["user_b_id"] == user_id)
#                     and m["score"] >= MATCH_THRESHOLD]

#     return templates.TemplateResponse(
#         "chats.html", 
#         {"request": request, "matches": user_matches, "current_user": user_id}
#     )

@app.get("/matches", response_class=HTMLResponse)
async def matches_page(request: Request):
    user_id = get_current_user(request)
    user_matches = [m for m in db["matches"]
                    if (m["user_a_id"] == user_id or m["user_b_id"] == user_id)
                    and m["score"] >= MATCH_THRESHOLD]

    return templates.TemplateResponse(
        "matches.html",
        {"request": request, "matches": user_matches, "current_user": user_id}
    )

# Optional single chat page (only if you have a template for it)

@app.get("/connections", response_class=HTMLResponse)
def connections_page(request: Request, with_user: Optional[str] = Query(None)):
    user_id = get_current_user(request)  # validates cookie/session

    connections = get_user_connections(user_id)
    # sort by name for consistent UI
    connections.sort(key=lambda c: c["name"].lower())

    active_user_id = None
    active_user_name = None

    if with_user and any(c["user_id"] == with_user for c in connections):
        active_user_id = with_user
        active_user_name = next(c["name"] for c in connections if c["user_id"] == with_user)
    elif connections:
        active_user_id = connections[0]["user_id"]
        active_user_name = connections[0]["name"]

    # fetch chat messages for the active connection (if any)
    messages = []
    if active_user_id:
        messages = [
            m for m in db.get("messages", [])
            if (m["sender"] == user_id and m["receiver"] == active_user_id)
            or (m["sender"] == active_user_id and m["receiver"] == user_id)
        ]
        messages.sort(key=lambda m: m.get("timestamp", ""))

    return templates.TemplateResponse(
        "connections.html",
        {
            "request": request,
            "current_user": user_id,
            "connections": connections,            # list of {user_id, name, score}
            "active_user_id": active_user_id,      # None if no connections
            "active_user_name": active_user_name,  # None if no connections
            "messages": messages
        }
    )

@app.post("/connections/send")
def connections_send(request: Request, to: str = Form(...), message: str = Form(...)):
    sender = get_current_user(request)

    # Ensure `to` is a matched connection
    allowed = {c["user_id"] for c in get_user_connections(sender)}
    if to not in allowed:
        raise HTTPException(status_code=403, detail="You can only chat with matched users.")

    db.setdefault("messages", []).append({
        "sender": sender,
        "receiver": to,
        "message": message.strip(),
        "timestamp": datetime.utcnow().isoformat()
    })
    save_data()
    # come back to the same conversation
    return RedirectResponse(url=f"/connections?with={to}", status_code=303)


@app.get("/chat/{chat_user}", response_class=HTMLResponse)
def chat_redirect(chat_user: str):
    # Optional: could validate session here if you want
    return RedirectResponse(url=f"/connections?with={chat_user}", status_code=303)

# PROFILE ROUTE
@app.get("/profile", response_class=HTMLResponse)
def profile(request: Request):
    print("✅ /profile route hit")
    user_id = request.cookies.get("user_id")

    if not user_id:
        return RedirectResponse(url="/login_page", status_code=303)

    # db["users"] is a list of dicts
    user = next((u for u in db["users"] if u["user_id"] == user_id), None)
    if not user:
        return RedirectResponse(url="/login_page", status_code=303)

    return templates.TemplateResponse(
        "profile.html",
        {"request": request, "user": user}
    )


# ------------------ AUTH -------------------------------

@app.post("/signup")
def signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    profession: str = Form(""),
    gender: str = Form(""),
    bio: str = Form("")
):
    uid = email.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="Missing email (user_id)")

    if any(x["user_id"] == uid for x in db["users"]):
        raise HTTPException(status_code=400, detail="User already exists")

    # Construct profile text for embedding
    profile_text = f"{profession} | {gender} | {bio}".strip()

    user = {
        "user_id": uid,               # primary key
        "display_name": name.strip(),
        "profile_text": profile_text,
        "password": password,
        "profession": profession,
        "gender": gender,
        "bio": bio
    }

    db["users"].append(user)
    cache_user_embedding(user)
    save_data()

    return RedirectResponse(url="/login_page", status_code=303)


@app.post("/login")
def login(
    response: Response,
    email: str = Form(...),
    password: str = Form(...)
):
    uid = email.strip()
    user = next((u for u in db["users"] if u["user_id"] == uid and u["password"] == password), None)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid.uuid4())
    sessions[token] = user["user_id"]

    resp = RedirectResponse(url="/feed", status_code=303)
    resp.set_cookie(key="session_token", value=token, httponly=True)
    return resp
# ------------------ POSTS & COMMENTS --------------------
@app.post("/add_post")
def add_post(request: Request, content: str = Form(...)):
    user_id = get_current_user(request)
    post_id = f"post{len(db['posts']) + 1}"
    db["posts"].append({
        "post_id": post_id,
        "user_id": user_id,
        "content": content.strip(),
        "created_at": now_iso()
    })
    save_data()
    return RedirectResponse(url="/feed", status_code=303)

@app.post("/add_comment")
def add_comment(request: Request, post_id: str = Form(...), text: str = Form(...)):
    user_id = get_current_user(request)
    if is_abusive(text):
        raise HTTPException(status_code=400, detail="Abusive or sensitive content detected")
    if not any(p["post_id"] == post_id for p in db["posts"]):
        raise HTTPException(status_code=400, detail="Post not found")

    comment_id = f"c{len(db['comments']) + 1}"
    new_comment = {
        "comment_id": comment_id,
        "user_id": user_id,
        "post_id": post_id,
        "text": text.strip(),
        "timestamp": now_iso()
    }
    db["comments"].append(new_comment)
    cache_comment_embedding(new_comment)  # ensures emb_cache AND JSON have the vector
    save_data()

    # immediately compute matches for this comment
    auto_match_for_comment(new_comment)

    return RedirectResponse(url="/feed", status_code=303)

# ------------------ DEBUG / JSON ------------------------
# @app.get("/matches")
# def get_matches():
#     return {"matches": db["matches"]}
@app.get("/matches")
def get_matches(request: Request):
    user_id = get_current_user(request)
    user_matches = [m for m in db["matches"]
                    if (m["user_a_id"] == user_id or m["user_b_id"] == user_id)
                    and m["score"] >= MATCH_THRESHOLD]
    return {"matches": user_matches}

@app.get("/debug_dump")
def dump_all():
    return JSONResponse(db)

# ------------------ INIT -------------------------------
load_data()
normalize_matches_schema()
rebuild_all_matches()
print("[INFO] Boot complete. Users:", len(db["users"]), "| Posts:", len(db["posts"]),
      "| Comments:", len(db["comments"]), "| Matches:", len(db["matches"]))
