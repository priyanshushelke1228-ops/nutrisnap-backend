#!/usr/bin/env python3
"""
NutriSnap AI - Production Secure Backend Server (Android & Web)
A zero-dependency production web server accessible from your Mac and Android phone,
with a secure backend proxy for Google Gemini Multimodal Vision AI.

Security:
- Credentials stored strictly in server-side environment variables (GEMINI_API_KEY).
- Client API keys are rejected; client localStorage keys are never read or stored.
- Payload and image size validation (max 5MB decoded image, 10MB HTTP payload).
- MIME type validation (image/jpeg, image/png, image/webp) and base64 corruption verification.
- Sliding-window rate limiting per client IP (15 requests / 60 seconds).
- 30-second upstream timeout with graceful error responses.
- Safe logging (no API keys, image payloads, or private data logged).
"""

import http.server
import socketserver
import os
import sys
import socket
import json
import time
import threading
import re
import hashlib
import base64
import urllib.request
import urllib.error
import urllib.parse
import webbrowser

PORT = int(os.environ.get("PORT", 8080))
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# In-memory thread-safe sliding rate limiter: { client_ip: [timestamp, ...] }
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 15
client_request_history = {}
rate_limit_lock = threading.Lock()

# Whitelist of supported Gemini multimodal models
SUPPORTED_MODELS = {
    "gemini-4-hydrogen": "gemini-4-hydrogen",     # Default: High-intelligence frontier multimodal vision
    "gemini-3.5-flash-lite": "gemini-3.5-flash-lite" # Efficiency: Low-latency fast multimodal vision
}
LEGACY_MODEL_MAP = {
    "gemini-1.5-flash": "gemini-4-hydrogen",
    "gemini-2.0-flash": "gemini-4-hydrogen",
    "gemini-1.5-pro": "gemini-4-hydrogen",
    "gemini-pro-vision": "gemini-4-hydrogen"
}
VISION_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-4-hydrogen"
]
NUTRIGUIDE_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-4-hydrogen"
]
TRANSIENT_HTTP_CODES = {404, 429, 500, 502, 503, 504}
EXCLUDED_DISCOVERED_TOKENS = (
    "embedding", "tts", "imagen", "veo", "lyria", "aqa",
    "audio", "live", "robotics", "computer-use", "deep-research", "image"
)
_discovered_models_cache = {"key_hash": "", "models": [], "expires_at": 0.0}
_discovered_models_lock = threading.Lock()


def build_model_candidates(primary_model, fallbacks):
    """Build an ordered, deduplicated list of candidate Gemini models starting with primary_model."""
    env_fallbacks = [
        item.strip() for item in os.environ.get("GEMINI_FALLBACK_MODELS", "").split(",") if item.strip()
    ]
    seen = set()
    ordered = []
    for m in [primary_model] + env_fallbacks + list(fallbacks or []):
        clean = (m or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered


def discover_gemini_models(api_key, timeout=5):
    """Query Google AI /v1beta/models (cached 10m) to discover available generateContent Gemini models."""
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _discovered_models_lock:
        if (
            _discovered_models_cache["key_hash"] == key_hash
            and _discovered_models_cache["expires_at"] > now
        ):
            return list(_discovered_models_cache["models"])

    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": api_key},
        method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    found = []
    for entry in (data.get("models") if isinstance(data, dict) else []) or []:
        if not isinstance(entry, dict):
            continue
        methods = entry.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        raw_name = str(entry.get("name") or "")
        model_id = raw_name.split("/", 1)[-1].strip()
        lower = model_id.lower()
        if not lower.startswith("gemini-"):
            continue
        if any(tok in lower for tok in EXCLUDED_DISCOVERED_TOKENS):
            continue
        found.append(model_id)

    def _rank(name):
        low = name.lower()
        if "flash-lite" in low:
            return (0, name)
        if "flash" in low:
            return (1, name)
        return (2, name)

    found.sort(key=_rank)
    with _discovered_models_lock:
        _discovered_models_cache["key_hash"] = key_hash
        _discovered_models_cache["models"] = list(found)
        _discovered_models_cache["expires_at"] = time.monotonic() + 600.0
    return found


def extract_gemini_text(gemini_data):
    """Extract only visible model answer text, skipping internal thought/reasoning parts."""
    if not isinstance(gemini_data, dict):
        return ""
    candidates = gemini_data.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return ""
    content = (candidates[0] or {}).get("content", {})
    parts = content.get("parts", []) if isinstance(content, dict) else []
    if not isinstance(parts, list):
        return ""
    return "".join(
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and not p.get("thought") and isinstance(p.get("text"), str)
    ).strip()


def _build_model_payload(gemini_payload, model_name, include_thinking_config=True):
    """Clone Gemini payload and attach safe minimal thinkingConfig appropriate for model_name."""
    cloned = dict(gemini_payload)
    gen_cfg = dict(gemini_payload.get("generationConfig") or {})
    if include_thinking_config and "thinkingConfig" not in gen_cfg:
        low = (model_name or "").lower()
        if low.startswith("gemini-2.5-flash"):
            gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
        elif low.startswith(("gemini-3", "gemini-4")):
            gen_cfg["thinkingConfig"] = {"thinkingLevel": "low"}
    cloned["generationConfig"] = gen_cfg
    return cloned


def call_gemini_with_fallback(
    api_key,
    candidate_models,
    gemini_payload,
    timeout=12,
    total_budget=42.0,
    max_retries_single=1,
    discover_on_exhaust=True
):
    """Call Gemini generateContent with automatic model failover, timeout budget, and live model discovery."""
    models = list(candidate_models or ["gemini-4-hydrogen"])
    deadline = time.monotonic() + float(total_budget)
    attempts_per_model = 1 + max_retries_single if len(models) == 1 else 1
    last_err = {"kind": "internal", "code": 500, "message": "Unknown Gemini upstream failure"}
    tried = set()
    discovered_checked = False

    def _try_model(model_name, max_attempts):
        nonlocal last_err
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        use_thinking_cfg = True
        for attempt in range(max_attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                return None, True
            req_timeout = min(float(timeout), max(2.0, remaining))
            payload_obj = _build_model_payload(gemini_payload, model_name, include_thinking_config=use_thinking_cfg)
            payload_bytes = json.dumps(payload_obj).encode('utf-8')
            req = urllib.request.Request(
                endpoint,
                data=payload_bytes,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key
                },
                method='POST'
            )
            try:
                with urllib.request.urlopen(req, timeout=req_timeout) as response:
                    result_bytes = response.read()
                parsed = json.loads(result_bytes.decode('utf-8'))
                if not isinstance(parsed, dict):
                    raise ValueError("Gemini API returned non-object JSON payload")
                if not extract_gemini_text(parsed):
                    last_err = {"kind": "http", "code": 502, "message": "Gemini model produced no visible text"}
                    return None, False
                return parsed, False
            except urllib.error.HTTPError as he:
                error_body = he.read().decode('utf-8', errors='ignore')
                err_msg = f"Gemini API HTTP {he.code}"
                try:
                    err_json = json.loads(error_body)
                    err_msg = err_json.get("error", {}).get("message", err_msg)
                except Exception:
                    if error_body:
                        err_msg = f"Gemini API error ({he.code}): {error_body[:200]}"
                if he.code == 400 and use_thinking_cfg and "thinking" in err_msg.lower():
                    # Model variant rejected thinkingConfig: retry once immediately without it
                    use_thinking_cfg = False
                    payload_obj = _build_model_payload(gemini_payload, model_name, include_thinking_config=False)
                    retry_req = urllib.request.Request(
                        endpoint,
                        data=json.dumps(payload_obj).encode('utf-8'),
                        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
                        method='POST'
                    )
                    try:
                        with urllib.request.urlopen(retry_req, timeout=req_timeout) as retry_resp:
                            retry_parsed = json.loads(retry_resp.read().decode('utf-8'))
                        if isinstance(retry_parsed, dict) and extract_gemini_text(retry_parsed):
                            return retry_parsed, False
                    except Exception:
                        pass
                last_err = {"kind": "http", "code": he.code, "message": err_msg}
                if he.code not in TRANSIENT_HTTP_CODES:
                    return None, True
                if attempt < max_attempts - 1 and (deadline - time.monotonic()) > 2.0:
                    time.sleep(0.5)
                    continue
                return None, False
            except (urllib.error.URLError, TimeoutError, socket.timeout) as ne:
                detail = getattr(ne, 'reason', None) or str(ne)
                last_err = {"kind": "network", "code": 504, "message": str(detail)}
                if attempt < max_attempts - 1 and (deadline - time.monotonic()) > 2.0:
                    time.sleep(0.5)
                    continue
                return None, False
            except Exception as ex:
                last_err = {"kind": "internal", "code": 500, "message": str(ex)}
                return None, True
        return None, False

    for model_name in models:
        tried.add(model_name)
        parsed, stop_now = _try_model(model_name, attempts_per_model)
        if parsed is not None:
            return parsed, model_name, None
        if stop_now:
            return None, model_name, last_err

    if discover_on_exhaust and not discovered_checked and (deadline - time.monotonic()) >= 4.0:
        discovered_checked = True
        extra_models = [
            m for m in discover_gemini_models(api_key, timeout=min(5.0, max(2.0, deadline - time.monotonic())))
            if m not in tried
        ][:3]
        for extra_model in extra_models:
            tried.add(extra_model)
            parsed, stop_now = _try_model(extra_model, 1)
            if parsed is not None:
                return parsed, extra_model, None
            if stop_now:
                return None, extra_model, last_err

    return None, models[-1], last_err

def get_local_ip():
    """Retrieve the primary local network IP address of this computer."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"

def _redact_secrets(text):
    """Ensure API keys never leak into exception or upstream error messages."""
    if not text:
        return ""
    return re.sub(r'(key=)[^&\s"\']+', r'\1REDACTED', str(text))

def is_rate_limited(client_ip):
    """Check if a client IP has exceeded the request threshold (thread-safe)."""
    now = time.time()
    with rate_limit_lock:
        if len(client_request_history) > 256:
            stale_ips = [
                ip for ip, timestamps in client_request_history.items()
                if not any(now - t < RATE_LIMIT_WINDOW_SECONDS for t in timestamps)
            ]
            for ip in stale_ips:
                del client_request_history[ip]
        history = client_request_history.get(client_ip, [])
        # Filter timestamps within the sliding window
        history = [t for t in history if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(history) >= RATE_LIMIT_MAX_REQUESTS:
            client_request_history[client_ip] = history
            return True
        history.append(now)
        client_request_history[client_ip] = history
        return False

YOUTH_SAFETY_REGEX = re.compile(
    r"\b(teen|teenager|teens|adolescent|minor|under\s*18|school\s*(?:kid|boy|girl|student)|(?:age|i am|i'm)\s*(?:1[0-7]|[89])\b)\b",
    re.IGNORECASE
)

def build_nutriguide_request(payload):
    """Validate and construct a safe, educational Gemini request payload for NutriGuide."""
    if not isinstance(payload, dict):
        return None, (400, "JSON payload must be an object.")

    raw_message = payload.get("message")
    if not isinstance(raw_message, str):
        return None, (400, "Message is required and must be a string.")
    message = raw_message.strip()
    if not message:
        return None, (400, "Message is required.")
    if len(message) > 2000:
        return None, (400, "Message exceeds maximum length of 2000 characters.")

    raw_history = payload.get("history")
    if raw_history is not None and not isinstance(raw_history, list):
        return None, (400, "Conversation history must be an array.")

    age_category = str(payload.get("ageCategory") or "").strip().lower()
    # Unknown or omitted age MUST default to safe educational mode; backend never weakens safety on client flags
    is_youth_safe = bool(payload.get("isYouthSafe", False)) or (age_category != "adult")
    user_profile = payload.get("userProfile")
    if isinstance(user_profile, dict):
        age = user_profile.get("age")
        if not isinstance(age, int) or isinstance(age, bool) or age < 18 or age > 120:
            is_youth_safe = True

    contents = []
    history_list = raw_history or []
    for h in history_list[-6:]:
        if not isinstance(h, dict):
            continue
        role = "user" if h.get("role") == "user" else "model"
        raw_t = h.get("text") if isinstance(h.get("text"), str) else h.get("content")
        text = raw_t.strip() if isinstance(raw_t, str) else ""
        if not text:
            continue
        if len(text) > 1000:
            text = text[:1000]
        if YOUTH_SAFETY_REGEX.search(text):
            is_youth_safe = True
        contents.append({
            "role": role,
            "parts": [{"text": text}]
        })

    if YOUTH_SAFETY_REGEX.search(message):
        is_youth_safe = True

    # Prevent duplicating the current user turn if already appended in history
    if contents and contents[-1]["role"] == "user" and contents[-1]["parts"][0]["text"] == message:
        contents.pop()

    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    # Base educational & anti-prescription guardrails enforced ALWAYS regardless of client age flags
    system_instruction = (
        "You are NutriGuide, a warm, culturally authentic, and scientifically grounded Indian food, culinary, and nutrition education assistant in NutriSnap AI.\n"
        "Your purpose is to help users explore Indian culinary traditions, ingredients, regional cuisines (Punjab, Maharashtra, Gujarat, South India, Bengal, Rajasthan, etc.), cooking techniques, recipe ideas, vegetarian protein alternatives, spice pairings, food storage, and reading packaged food nutrition labels.\n\n"
        "STRICT PRODUCT & SAFETY MANDATES (ENFORCED FOR ALL USERS INDEPENDENTLY OF AGE FLAGS):\n"
        "1. EDUCATIONAL ONLY: You are a culinary and nutrition educator, NEVER a medical doctor, dietitian, or clinician. You MUST NOT diagnose medical conditions or prescribe clinical treatment.\n"
        "2. NO DIET PRESCRIPTIONS OR CALORIE RESTRICTION FOR ANYONE: Regardless of user age or request framing, NEVER prescribe personal daily calorie targets, calorie deficits, restrictive eating regimens, crash diets, meal skipping, or fasting plans, and NEVER rate bodies or prescribe weight loss.\n"
        "3. ALLERGEN CAUTION: Never guarantee that any food, homemade dish, or packaged product is 100% allergen-safe. Always advise checking actual packaging and manufacturer labels for potential allergen cross-contamination.\n"
        "4. ACCURACY: Never fabricate fake nutritional facts, invented statistics, or unverified claims. Base Indian food context on standard ICMR-NIN composition knowledge.\n"
        "5. LANGUAGE: Communicate exclusively in clear, accessible English.\n"
        "6. NON-JUDGMENTAL: Never label traditional foods as 'sinful', 'guilt-ridden', or 'bad'.\n"
        "7. RESPONSE FORMAT: Reply directly to the user in a clear, concise, practical way (~120-260 words unless more detail is requested) using short paragraphs and clean bullet points. NEVER output meta-commentary, rule checklists, or self-verification notes about your instructions.\n"
    )

    if is_youth_safe:
        system_instruction += (
            "\nCRITICAL SAFE EDUCATIONAL & YOUTH SAFETY POLICY (ACTIVE):\n"
            "- The user is either under 18 or has not confirmed adult age (default safe educational mode).\n"
            "- Focus strictly on cooking enjoyment, ingredients, balanced meal appreciation, kitchen skills, and cultural recipes.\n"
            "- Absolutely DO NOT calculate, recommend, or discuss personal calorie targets, calorie deficits, restrictive diets, macro caps, or weight-loss plans.\n"
            "- If the user asks about losing weight or restricting calories, kindly explain that growing bodies need steady, nourishing fuel, and suggest discussing health and growth with a pediatrician, doctor, or guardian.\n"
        )

    gemini_payload = {
        "contents": contents,
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 2500
        }
    }
    return {"gemini_payload": gemini_payload, "is_youth_safe": is_youth_safe}, None

class NutriSnapHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def get_client_ip(self):
        """Resolve real caller IP behind Render/Cloudflare proxies before falling back to socket address."""
        hdrs = getattr(self, 'headers', None)
        if hdrs:
            fwd = hdrs.get('X-Forwarded-For', '')
            if fwd:
                first = fwd.split(',')[0].strip()
                if first:
                    return first[:64]
            real_ip = hdrs.get('X-Real-IP', '')
            if real_ip and real_ip.strip():
                return real_ip.strip()[:64]
        return self.client_address[0] if self.client_address else "unknown"

    def send_json(self, status_code, data, include_body=True):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def send_json_error(self, status_code, message, upstream_code=None, error_type=None, include_body=True):
        err_obj = {
            "code": upstream_code if upstream_code is not None else status_code,
            "message": message
        }
        if error_type:
            err_obj["errorType"] = error_type
        self.send_json(status_code, {"error": err_obj}, include_body=include_body)

    def _send_upstream_gemini_error(self, err_info, context_label):
        """Map structured upstream Gemini failure into safe, redacted JSON HTTP error."""
        kind = (err_info or {}).get("kind")
        msg = _redact_secrets((err_info or {}).get("message", "Unknown upstream error"))
        if kind == "http":
            code = err_info.get("code", 502)
            out_status = code if code in [400, 401, 403, 429] else 502
            err_type = "api_auth_error" if code in [401, 403] else ("api_quota_error" if code == 429 else "api_error")
            self.send_json_error(out_status, msg, upstream_code=code, error_type=err_type)
            return
        if kind == "network":
            self.send_json_error(504, f"Network error contacting {context_label}: {msg}", error_type="network_error")
            return
        self.send_json_error(500, f"Internal {context_label} error: {msg}", error_type="server_error")

    def _read_json_object(self, max_bytes, too_large_msg):
        raw_len = self.headers.get('Content-Length')
        if raw_len is None:
            self.send_json_error(400, "Missing Content-Length header.")
            return None
        try:
            content_length = int(raw_len)
        except (ValueError, TypeError):
            self.send_json_error(400, "Invalid Content-Length header.")
            return None

        if content_length <= 0:
            self.send_json_error(400, "Request body must not be empty.")
            return None
        if content_length > max_bytes:
            self.send_json_error(413, too_large_msg)
            return None

        try:
            body_bytes = self.rfile.read(content_length)
            payload = json.loads(body_bytes.decode('utf-8'))
        except Exception as e:
            self.send_json_error(400, f"Invalid JSON payload: {str(e)}")
            return None

        if not isinstance(payload, dict):
            self.send_json_error(400, "JSON payload must be an object.")
            return None
        return payload

    def end_headers(self):
        # Enable CORS, security, and caching headers for smooth mobile / Android WebView experience
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-NutriSnap-App')
        self.send_header('Access-Control-Max-Age', '86400')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Service-Worker-Allowed', '/')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _is_backend_only(self):
        """Return True when deployed without web frontend assets (no index.html)."""
        base_dir = getattr(self, 'directory', None) or DIRECTORY
        return not os.path.isfile(os.path.join(base_dir, 'index.html'))

    def _is_blocked_static_path(self, path_only):
        """Block dotfiles (.git, .env, .venv) and keystore files from static serving."""
        decoded = urllib.parse.unquote(path_only or '')
        segments = [seg for seg in decoded.replace('\\', '/').split('/') if seg]
        for seg in segments:
            if seg.startswith('.') or seg.lower().endswith(('.keystore', '.pem')):
                return True
        return False

    def _send_backend_status_page(self, include_body=True):
        """Serve a clean status page (or JSON if requested) at / in backend-only mode."""
        has_env_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        hdrs = getattr(self, 'headers', None)
        accept = (hdrs.get('Accept', '') if hdrs else '').lower()

        if 'application/json' in accept and 'text/html' not in accept:
            payload = {
                "service": "NutriSnap AI Backend",
                "status": "ok",
                "mode": "backend-only",
                "version": "1.5.3",
                "hasBackendKey": has_env_key,
                "defaultModel": "gemini-4-hydrogen",
                "endpoints": [
                    "/api/health",
                    "/api/config",
                    "/api/analyze-meal",
                    "/api/scan-label",
                    "/api/nutriguide"
                ]
            }
            self.send_json(200, payload, include_body=include_body)
            return

        key_badge = (
            '<span class="badge ok">GEMINI_API_KEY Active</span>'
            if has_env_key
            else '<span class="badge warn">GEMINI_API_KEY Not Set</span>'
        )
        html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NutriSnap AI — Backend API</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{
      margin: 0; font-family: system-ui, -apple-system, sans-serif;
      background: #0f172a; color: #e2e8f0;
      display: grid; place-items: center; min-height: 100vh; padding: 24px;
    }}
    .card {{
      max-width: 560px; width: 100%; background: #1e293b;
      border: 1px solid #334155; border-radius: 16px; padding: 28px;
      box-shadow: 0 20px 45px rgba(0, 0, 0, 0.35);
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.35rem; color: #f8fafc; }}
    p {{ margin: 0 0 18px; color: #94a3b8; line-height: 1.5; font-size: 0.95rem; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
    .badge {{
      display: inline-block; padding: 4px 10px; border-radius: 999px;
      font-size: 0.78rem; font-weight: 600; background: #334155; color: #e2e8f0;
    }}
    .badge.ok {{ background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }}
    .badge.warn {{ background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }}
    ul {{ list-style: none; padding: 0; margin: 0; display: grid; gap: 8px; }}
    li {{
      display: flex; justify-content: space-between; gap: 12px; align-items: center;
      background: #0f172a; border: 1px solid #334155; border-radius: 10px;
      padding: 10px 12px; font-size: 0.88rem;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, monospace; color: #38bdf8; }}
    a {{ color: #38bdf8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .method {{ font-size: 0.75rem; font-weight: 700; color: #94a3b8; }}
  </style>
</head>
<body>
  <main class="card">
    <h1>NutriSnap AI — Backend API</h1>
    <p>This service runs in <strong>backend-only</strong> mode for the NutriSnap mobile and web clients. No web UI or directory listing is served at <code>/</code>.</p>
    <div class="row">
      <span class="badge ok">API Online · v1.5.1</span>
      {key_badge}
    </div>
    <ul>
      <li><a href="/api/health"><code>/api/health</code></a><span class="method">GET</span></li>
      <li><a href="/api/config"><code>/api/config</code></a><span class="method">GET</span></li>
      <li><code>/api/analyze-meal</code><span class="method">POST</span></li>
      <li><code>/api/scan-label</code><span class="method">POST</span></li>
      <li><code>/api/nutriguide</code><span class="method">POST</span></li>
    </ul>
  </main>
</body>
</html>"""
        body = html_doc.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def list_directory(self, path):
        """Never expose raw filesystem directory listings."""
        self.send_json_error(404, "Directory listing is disabled.")
        return None

    def _serve_api_get(self, path_only, include_body=True):
        # Endpoint to return the server's LAN IP for the Android QR code modal
        if path_only == '/api/server-info':
            local_ip = get_local_ip()
            self.send_json(200, {
                "localIp": local_ip,
                "port": PORT,
                "wifiUrl": f"http://{local_ip}:{PORT}"
            }, include_body=include_body)
            return True

        # Endpoint to report backend capability, health, and status
        if path_only in ['/api/config', '/api/health']:
            has_env_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
            env_vision_model = os.environ.get("GEMINI_VISION_MODEL", "").strip()
            configured_model = env_vision_model or "gemini-4-hydrogen"
            model_config_valid = (
                configured_model in SUPPORTED_MODELS or configured_model in LEGACY_MODEL_MAP
            )
            default_model = (
                SUPPORTED_MODELS.get(configured_model)
                or LEGACY_MODEL_MAP.get(configured_model)
                or configured_model
            )
            self.send_json(200, {
                "status": "healthy" if path_only == '/api/health' else "ok",
                "healthy": True,
                "version": "1.5.3",
                "hasBackendKey": bool(has_env_key),
                "modelConfigValid": bool(model_config_valid),
                "defaultModel": default_model,
                "supportedModels": list(SUPPORTED_MODELS.keys()),
                "rateLimit": f"{RATE_LIMIT_MAX_REQUESTS} req/{RATE_LIMIT_WINDOW_SECONDS}s"
            }, include_body=include_body)
            return True

        return False

    def _handle_static_or_backend_root(self, include_body=True):
        """Route GET and HEAD consistently across API endpoints, backend-only root, and local static files."""
        path_only = self.path.split('?')[0]
        if self._serve_api_get(path_only, include_body=include_body):
            return

        if self._is_blocked_static_path(path_only):
            self.send_json_error(404, "Not found.", include_body=include_body)
            return

        if self._is_backend_only():
            if path_only in ['', '/']:
                self._send_backend_status_page(include_body=include_body)
                return
            self.send_json_error(
                404,
                "Not found. This server runs in backend-only mode with API endpoints under /api/*.",
                include_body=include_body
            )
            return

        if include_body:
            super().do_GET()
        else:
            super().do_HEAD()

    def do_HEAD(self):
        self._handle_static_or_backend_root(include_body=False)

    def do_GET(self):
        self._handle_static_or_backend_root(include_body=True)

    def do_POST(self):
        path_only = self.path.split('?')[0]
        if path_only in ['/api/analyze-meal', '/api/scan-label']:
            client_ip = self.get_client_ip()

            # 1. Sliding-window Rate limiting check
            if is_rate_limited(client_ip):
                self.send_json_error(
                    429,
                    f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS} AI meal scans per minute. Please wait.",
                    error_type="rate_limit_error"
                )
                return

            # 2. Payload size & schema guard (max 10MB, JSON object required)
            payload = self._read_json_object(
                10 * 1024 * 1024,
                "Payload too large. Meal photo request must be under 10MB."
            )
            if payload is None:
                return

            # 3. Server-side Credentials Verification
            # Security Rule: Client API keys are rejected. Secrets must live in server-side environment variables.
            active_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not active_key:
                self.send_json_error(
                    503,
                    "Production Gemini API key is not configured on the server. Please set the GEMINI_API_KEY environment variable on the backend server.",
                    error_type="server_config_error"
                )
                return

            # 4. Extract and validate image payload
            image_raw = payload.get("imageBase64", "")
            if not image_raw or not isinstance(image_raw, str):
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"code": 400, "message": "imageBase64 is required and must be a valid base64 data URL string"}
                }).encode('utf-8'))
                return

            mime_type = "image/jpeg"
            clean_base64 = image_raw
            b64_marker_idx = image_raw.lower().find("base64,")
            if b64_marker_idx != -1:
                header_part = image_raw[:b64_marker_idx + len("base64,")]
                clean_base64 = image_raw[b64_marker_idx + len("base64,"):]
                header_lower = header_part.lower()
                if "image/png" in header_lower:
                    mime_type = "image/png"
                elif "image/webp" in header_lower:
                    mime_type = "image/webp"
                elif "image/jpeg" in header_lower or "image/jpg" in header_lower:
                    mime_type = "image/jpeg"
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": {"code": 400, "message": "Unsupported image format. Allowed formats: JPEG, PNG, WebP."}
                    }).encode('utf-8'))
                    return

            # Strip whitespace and validate base64 integrity and decoded byte size (1B..5MB)
            clean_base64 = "".join(clean_base64.split())
            if not clean_base64:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"code": 400, "message": "Empty image data provided."}
                }).encode('utf-8'))
                return

            try:
                decoded_image = base64.b64decode(clean_base64, validate=True)
                if len(decoded_image) == 0:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": {"code": 400, "message": "Empty image data provided."}
                    }).encode('utf-8'))
                    return
                if len(decoded_image) > 5 * 1024 * 1024:
                    self.send_response(413)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": {"code": 413, "message": "Image exceeds maximum allowed size of 5MB."}
                    }).encode('utf-8'))
                    return
            except Exception as b64_err:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"code": 400, "message": f"Malformed base64 image data: {str(b64_err)}"}
                }).encode('utf-8'))
                return

            # 5. Model Validation & Migration
            raw_model = payload.get("model")
            requested_model = (raw_model.strip() if isinstance(raw_model, str) else "") or "gemini-4-hydrogen"
            if requested_model in LEGACY_MODEL_MAP:
                selected_model = LEGACY_MODEL_MAP[requested_model]
            elif requested_model in SUPPORTED_MODELS:
                selected_model = SUPPORTED_MODELS[requested_model]
            else:
                selected_model = "gemini-4-hydrogen"

            # 6. Structured Multimodal Food Analysis Instruction
            raw_scan_type = payload.get("scanType")
            scan_type = (raw_scan_type.strip() if isinstance(raw_scan_type, str) else "") or (
                "nutrition_label" if path_only == '/api/scan-label' else "meal"
            )

            if scan_type == "nutrition_label":
                system_instruction = (
                    "You are an expert OCR and packaged food nutrition facts label analyst.\n"
                    "Inspect this photo of a food package, nutrition facts table, or ingredients label.\n\n"
                    "1. UNREADABLE / BLURRY / NON-PACKAGING DETECTION:\n"
                    "If the image is too blurry, unreadable, dark, or does not contain a nutrition label or ingredient list:\n"
                    "Return JSON with: isLabel: false, isFood: false, title: \"Unreadable Label\", "
                    "error: \"The label text is unreadable or poorly lit. Please retake a clear, well-lit photo of the nutrition facts table.\", "
                    "uncertainFields: [\"all\"]\n\n"
                    "2. PRECISE NUTRITION FACTS & MEASUREMENT BASIS:\n"
                    "- Identify the primary measurement basis: \"per_100g\", \"per_100ml\", or \"per_serving\".\n"
                    "- Extract servingSize (e.g. \"30g\", \"250ml\", \"1 pouch (45g)\"), servingUnit (\"g\", \"ml\", or \"portion\"), servingAmount (number or null).\n"
                    "- NEVER assume millilitres equal grams.\n"
                    "- Extract available nutrition values: energy/calories (kcal), protein (g), carbohydrates (g), addedSugars (g), total fat (g), saturated fat (g), trans fat (g), fiber (g), sodium (mg).\n"
                    "- If a nutrient is NOT declared or unreadable on the label, set it to null. NEVER fabricate or substitute zero for missing values.\n"
                    "- If any extracted value is blurry, partially obscured, or uncertain, add the field name to \"uncertainFields\" array (e.g. [\"protein\", \"addedSugars\"]).\n\n"
                    "3. INGREDIENTS, ALLERGENS & STORAGE:\n"
                    "- Extract ingredients text verbatim.\n"
                    "- Extract declared allergens list and possible traces list.\n"
                    "- If allergens are not declared or not visible, set declaredAllergens: [], possibleTraces: [], allergenStatus: \"unavailable\".\n"
                    "- Extract storage instructions if visible (e.g. \"Store in a cool dry place\"), otherwise null.\n\n"
                    "Return PURE JSON in this schema:\n"
                    "{\n"
                    "  \"isLabel\": true,\n"
                    "  \"isFood\": true,\n"
                    "  \"title\": \"Extracted Product Name\",\n"
                    "  \"brand\": \"Extracted Brand\",\n"
                    "  \"servingSize\": \"30g\",\n"
                    "  \"servingUnit\": \"g\",\n"
                    "  \"servingAmount\": 30,\n"
                    "  \"measurementBasis\": \"per_100g\",\n"
                    "  \"cal\": 150,\n"
                    "  \"p\": 3.5,\n"
                    "  \"c\": 22.0,\n"
                    "  \"f\": 5.0,\n"
                    "  \"fiber\": 1.2,\n"
                    "  \"addedSugars\": 4.0,\n"
                    "  \"sodium\": 120,\n"
                    "  \"ingredients\": \"Extracted ingredients text...\",\n"
                    "  \"declaredAllergens\": [\"Wheat\"],\n"
                    "  \"possibleTraces\": [\"Milk\"],\n"
                    "  \"allergenStatus\": \"declared\",\n"
                    "  \"storageInstructions\": \"Store in a cool, dry place\",\n"
                    "  \"uncertainFields\": [],\n"
                    "  \"notes\": \"Packaging label extraction\"\n"
                    "}"
                )
            else:
                system_instruction = (
                    "You are an expert multimodal nutritionist and vision analyst specializing in Indian and global cuisine.\n"
                    "Inspect this meal photo carefully.\n\n"
                    "1. NON-FOOD DETECTION:\n"
                    "If the image does not show food or beverages (e.g. phones, people, furniture, documents, electronics, animals):\n"
                    "Return JSON with: isFood: false, title: \"Non-food item detected\", category: \"Other\", items: [], totals: {cal:0, p:0, c:0, f:0, fiber:0}.\n"
                    "CRITICAL: items MUST strictly be an empty array [] if isFood is false. Never return contradictory food items for non-food images.\n\n"
                    "2. WATER & HYDRATION DETECTION:\n"
                    "If the photo shows only water (a glass of plain water, mineral water bottle, hydration flask):\n"
                    "Return JSON with: isWater: true, title: \"Hydration / Water Bottle\", category: \"Beverage\", items: [{name: \"Water\", servingCount: 1, servingUnit: \"glass (500ml)\", weight: 500, unit: \"ml (500ml)\", cal: 0, p: 0, c: 0, f: 0, fiber: 0, box_2d: null}], totals: {cal:0, p:0, c:0, f:0, fiber: 0}.\n\n"
                    "3. ACCURATE DISH IDENTIFICATION, SMART THALI LOCALIZATION & PORTIONS:\n"
                    "Identify each distinct food item on the plate.\n"
                    "For each food item:\n"
                    "- Separate the serving count (e.g. 2, 1, 0.5), household unit (e.g. \"piece\", \"katori\", \"bowl\", \"cup\", \"tbsp\", \"plate\"), and estimated portion weight in grams.\n"
                    "- Never set weight to piece count (e.g. 2 pieces must NOT be 2g; estimated weight for 2 phulkas is ~60g).\n"
                    "- If nutritional values or weight are unknown, set them to null. Never output negative numbers or fabricated figures.\n"
                    "- Smart Thali localization: If the dish boundary on the plate/image is clearly visible, provide normalized bounding box coordinates box_2d: [ymin, xmin, ymax, xmax] on an integer scale of 0 to 1000 (e.g. [120, 150, 480, 520]). If the region cannot be reliably determined, set box_2d: null.\n"
                    "Always set coordinate_format: \"normalized_0_to_1000\" in root JSON.\n"
                    "Never invent confidence scores. Only include confidence as a genuine float 0.0-1.0 if highly certain; otherwise set it to null.\n"
                    "Always return PURE JSON matching this schema:\n"
                    "{\n"
                    "  \"title\": \"Descriptive Meal Name\",\n"
                    "  \"category\": \"Breakfast | Lunch | Dinner | Snack\",\n"
                    "  \"coordinate_format\": \"normalized_0_to_1000\",\n"
                    "  \"confidence\": null,\n"
                    "  \"dietaryTags\": [\"Pure Veg\", \"High Protein\", etc.],\n"
                    "  \"notes\": \"Nutritional insights and advice\",\n"
                    "  \"isFood\": true,\n"
                    "  \"isWater\": false,\n"
                    "  \"items\": [\n"
                    "    {\n"
                    "      \"name\": \"Whole Wheat Phulka / Roti (No Ghee)\",\n"
                    "      \"servingCount\": 2,\n"
                    "      \"servingUnit\": \"piece\",\n"
                    "      \"weight\": 60,\n"
                    "      \"unit\": \"2 pieces (60g)\",\n"
                    "      \"cal\": 150,\n"
                    "      \"p\": 6.2,\n"
                    "      \"c\": 30.4,\n"
                    "      \"f\": 0.8,\n"
                    "      \"fiber\": 4.6,\n"
                    "      \"box_2d\": [120, 100, 450, 420]\n"
                    "    }\n"
                    "  ]\n"
                    "}"
                )

            gemini_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": system_instruction},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": clean_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.8,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json"
                }
            }

            candidate_models = build_model_candidates(selected_model, VISION_FALLBACK_MODELS)
            gemini_response, used_model, err_info = call_gemini_with_fallback(
                active_key,
                candidate_models,
                gemini_payload,
                timeout=12
            )
            if err_info is not None:
                self._send_upstream_gemini_error(err_info, "Gemini API")
                return

            try:
                raw_text = extract_gemini_text(gemini_response)
                if not raw_text:
                    raise ValueError("Empty response received from Gemini model")

                clean_text = raw_text.strip()
                if clean_text.startswith("```json"):
                    clean_text = clean_text[7:]
                if clean_text.startswith("```"):
                    clean_text = clean_text[3:]
                if clean_text.endswith("```"):
                    clean_text = clean_text[:-3]
                clean_text = clean_text.strip()

                parsed_result = json.loads(clean_text)
                if not isinstance(parsed_result, dict):
                    raise ValueError("Gemini model returned non-object JSON payload")
                parsed_result["source"] = f"Gemini Vision Proxy ({used_model})"

                if scan_type == "nutrition_label":
                    parsed_result["scanType"] = "nutrition_label"
                    parsed_result["isUserConfirmed"] = False
                    parsed_result["allergenNotice"] = "⚠️ Allergen Notice: Allergen information is extracted from packaging text. Always check physical packaging before consumption. NutriSnap cannot guarantee allergen safety."
                    if not isinstance(parsed_result.get("uncertainFields"), list):
                        parsed_result["uncertainFields"] = []
                else:
                    # Ensure confidence integrity: strip fake or invalid confidence numbers
                    raw_conf = parsed_result.get("confidence")
                    if not isinstance(raw_conf, bool) and isinstance(raw_conf, (int, float)) and 0.0 <= raw_conf <= 1.0:
                        parsed_result["confidence"] = round(float(raw_conf), 2)
                    else:
                        parsed_result["confidence"] = None

                    # Ensure coordinate_format is present
                    if "coordinate_format" not in parsed_result:
                        parsed_result["coordinate_format"] = "normalized_0_to_1000"

                    # Ensure items array structure
                    if not isinstance(parsed_result.get("items"), list):
                        parsed_result["items"] = []

                self.send_json(200, parsed_result)
                return
            except Exception as parse_err:
                self.send_json_error(
                    500,
                    f"Internal meal analysis error: {_redact_secrets(parse_err)}",
                    error_type="server_error"
                )
                return

        elif path_only == '/api/nutriguide':
            client_ip = self.get_client_ip()

            # 1. Sliding-window Rate limiting check
            if is_rate_limited(client_ip):
                self.send_json_error(
                    429,
                    f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS} requests per minute. Please wait.",
                    error_type="rate_limit_error"
                )
                return

            # 2. Payload size & JSON object guard (max 64KB for text Q&A)
            payload = self._read_json_object(
                64 * 1024,
                "Payload too large. NutriGuide message must be under 64KB."
            )
            if payload is None:
                return

            # 3. Server-side Credentials Verification
            active_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not active_key:
                self.send_json_error(
                    503,
                    "Production Gemini API key is not configured on the server. Please set the GEMINI_API_KEY environment variable on the backend server.",
                    error_type="server_config_error"
                )
                return

            # 4-6. Strict message, history, and safety validation
            built_req, err_tuple = build_nutriguide_request(payload)
            if err_tuple is not None:
                err_code, err_msg = err_tuple
                self.send_json_error(err_code, err_msg, error_type="validation_error")
                return

            gemini_payload = built_req["gemini_payload"]
            is_youth_safe = built_req["is_youth_safe"]

            primary_guide_model = os.environ.get("GEMINI_NUTRIGUIDE_MODEL", "").strip() or "gemini-3.8-flash"
            candidate_guide_models = build_model_candidates(primary_guide_model, NUTRIGUIDE_FALLBACK_MODELS)

            gemini_data, used_model, err_info = call_gemini_with_fallback(
                active_key,
                candidate_guide_models,
                gemini_payload,
                timeout=12
            )
            if err_info is not None:
                self._send_upstream_gemini_error(err_info, "NutriGuide assistant")
                return

            reply_text = extract_gemini_text(gemini_data)
            if not reply_text:
                reply_text = "I'm sorry, I couldn't generate a response. Please try asking your question differently."
            elif len(reply_text) > 12000:
                reply_text = reply_text[:12000]

            self.send_json(200, {
                "status": "ok",
                "reply": reply_text,
                "isYouthSafe": is_youth_safe,
                "safetyEnforcedIndependently": True,
                "model": used_model
            })
            return

        self.send_response(404)
        self.end_headers()

    def guess_type(self, path):
        # Ensure correct MIME types for web files and APK
        if path.endswith('.js'):
            return 'application/javascript'
        elif path.endswith('.css'):
            return 'text/css'
        elif path.endswith('.html'):
            return 'text/html'
        elif path.endswith('.json') or path.endswith('.webmanifest'):
            return 'application/manifest+json' if 'manifest' in path else 'application/json'
        elif path.endswith('.svg'):
            return 'image/svg+xml'
        elif path.endswith('.apk'):
            return 'application/vnd.android.package-archive'
        elif path.endswith('.aab'):
            return 'application/octet-stream'
        return super().guess_type(path)

class ThreadingNutriSnapServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def run():
    global PORT
    os.chdir(DIRECTORY)
    local_ip = get_local_ip()

    server = None
    for attempt_port in range(PORT, PORT + 20):
        try:
            server = ThreadingNutriSnapServer(("0.0.0.0", attempt_port), NutriSnapHandler)
            PORT = attempt_port
            break
        except OSError:
            continue

    if not server:
        print(f"❌ Error: Could not bind to any port between {PORT} and {PORT+20}")
        sys.exit(1)

    local_url = f"http://localhost:{PORT}"
    android_url = f"http://{local_ip}:{PORT}"

    has_backend_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    key_status = "✅ ACTIVE (Server ENV)" if has_backend_key else "⚠️ NOT SET (Export GEMINI_API_KEY)"

    print("=" * 64)
    print("🥗  NutriSnap AI - Production Backend Server")
    print("=" * 64)
    print(f"💻 Mac / Browser:   {local_url}")
    print(f"📱 Android Phone:   {android_url}")
    print(f"🔑 Gemini Backend:  {key_status}")
    print(f"🤖 Vision Model:   gemini-4-hydrogen (Default)")
    print("-" * 64)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user.")
        server.server_close()

if __name__ == '__main__':
    run()
