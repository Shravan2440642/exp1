"""
Jarvis WebSocket Gateway — Multi-LLM Support
Supports: Claude, OpenAI (GPT), Google Gemini, Groq, Ollama (local)

Install (pick what you need):
  pip install websockets anthropic openai google-genai

Run:
  export ANTHROPIC_API_KEY=sk-ant-...    # for Claude
  export OPENAI_API_KEY=sk-...           # for GPT
  export GEMINI_API_KEY=AIza...          # for Gemini
  export GROQ_API_KEY=gsk_...            # for Groq (uses OpenAI SDK)

  python gateway.py                       # auto-detects your key
  python gateway.py --provider gemini     # force a specific provider
  python gateway.py --provider ollama     # free local models
"""

import asyncio
import json
import time
import uuid
import os
import sys
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

HOST = "127.0.0.1"
PORT = 18789
MAX_HISTORY = 20

SYSTEM_PROMPT = """You are Jarvis, a personal AI assistant inspired by 
Iron Man's Jarvis. You are helpful, concise, and proactive. You speak 
in a calm, professional tone with occasional dry wit. 
Address the user as "sir" occasionally for flavor.
Keep responses concise — 2-3 sentences for simple questions."""


# ═══════════════════════════════════════════════════════════════
# LLM PROVIDERS — add your own by subclassing LLMProvider
# ═══════════════════════════════════════════════════════════════

class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def chat(self, messages: list, system: str) -> dict:
        """Returns {"text": str, "input_tokens": int, "output_tokens": int}"""
        pass


# ── Claude (Anthropic) ────────────────────────────────────────

class ClaudeProvider(LLMProvider):
    name = "Claude"

    def __init__(self, model="claude-sonnet-4-20250514"):
        from anthropic import AsyncAnthropic
        self.model = model
        self.client = AsyncAnthropic()

    async def chat(self, messages: list, system: str) -> dict:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }


# ── OpenAI (GPT-4o, GPT-4o-mini, etc.) ───────────────────────

class OpenAIProvider(LLMProvider):
    name = "OpenAI"

    def __init__(self, model="gpt-4o-mini"):
        from openai import AsyncOpenAI
        import httpx
        self.model = model
        self.client = AsyncOpenAI(
            http_client=httpx.AsyncClient(verify=False),
        )

    async def chat(self, messages: list, system: str) -> dict:
        full_messages = [{"role": "system", "content": system}] + messages
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=full_messages,
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }


# ── Google Gemini ─────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    name = "Gemini"

    def __init__(self, model="gemini-2.0-flash"):
        from google import genai
        self.model = model
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )

    async def chat(self, messages: list, system: str) -> dict:
        # convert messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model,
            contents=contents,
            config={
                "system_instruction": system,
                "max_output_tokens": 1024,
            },
        )
        usage = response.usage_metadata
        return {
            "text": response.text,
            "input_tokens": usage.prompt_token_count if usage else 0,
            "output_tokens": usage.candidates_token_count if usage else 0,
        }


# ── Groq (fast inference — Llama, Mixtral, etc.) ──────────────

class GroqProvider(LLMProvider):
    name = "Groq"

    def __init__(self, model="llama-3.3-70b-versatile"):
        from openai import AsyncOpenAI
        import httpx
        self.model = model
        self.client = AsyncOpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            http_client=httpx.AsyncClient(verify=False),
        )

    async def chat(self, messages: list, system: str) -> dict:
        full_messages = [{"role": "system", "content": system}] + messages
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=full_messages,
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }


# ── Ollama (free, local models) ───────────────────────────────

class OllamaProvider(LLMProvider):
    name = "Ollama"

    def __init__(self, model="llama3.2"):
        from openai import AsyncOpenAI
        import httpx
        self.model = model
        self.client = AsyncOpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            http_client=httpx.AsyncClient(verify=False),
        )

    async def chat(self, messages: list, system: str) -> dict:
        full_messages = [{"role": "system", "content": system}] + messages
        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=full_messages,
        )
        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
        }


# ═══════════════════════════════════════════════════════════════
# AUTO-DETECT PROVIDER
# ═══════════════════════════════════════════════════════════════

def detect_provider(forced: str = None) -> LLMProvider:
    """Auto-detect which provider to use based on env vars."""

    providers = {
        "claude":  ("ANTHROPIC_API_KEY", ClaudeProvider,  "claude-sonnet-4-20250514"),
        "openai":  ("OPENAI_API_KEY",   OpenAIProvider,   "gpt-4o-mini"),
        "gemini":  ("GEMINI_API_KEY",   GeminiProvider,   "gemini-2.0-flash"),
        "groq":    ("GROQ_API_KEY",     GroqProvider,     "llama-3.3-70b-versatile"),
        "ollama":  (None,               OllamaProvider,   "llama3.2"),
    }

    # forced provider
    if forced:
        forced = forced.lower()
        if forced in providers:
            env_key, cls, model = providers[forced]
            if env_key and not os.environ.get(env_key):
                print(f"  [!] Warning: {env_key} not set for {forced}")
            return cls(model)
        else:
            print(f"  [!] Unknown provider: {forced}")
            print(f"  Available: {', '.join(providers.keys())}")
            sys.exit(1)

    # auto-detect from env vars (priority order)
    for name, (env_key, cls, model) in providers.items():
        if env_key is None:
            continue
        if os.environ.get(env_key):
            print(f"  [*] Auto-detected: {name} ({env_key} found)")
            return cls(model)

    # fallback to Ollama (no API key needed)
    print("  [*] No API keys found — falling back to Ollama (local)")
    print("  [*] Make sure Ollama is running: ollama serve")
    return OllamaProvider()


# ═══════════════════════════════════════════════════════════════
# SESSION MODEL
# ═══════════════════════════════════════════════════════════════

@dataclass
class Session:
    id: str
    peer_id: str
    messages: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    message_count: int = 0

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.message_count += 1
        if len(self.messages) > MAX_HISTORY:
            self.messages = self.messages[-MAX_HISTORY:]

    def reset(self):
        self.messages.clear()
        self.message_count = 0
        self.created_at = time.time()


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def get_or_create(self, peer_id: str) -> Session:
        if peer_id not in self.sessions:
            self.sessions[peer_id] = Session(
                id=str(uuid.uuid4())[:8],
                peer_id=peer_id,
            )
            print(f"  [+] New session: {peer_id}")
        return self.sessions[peer_id]

    def get_stats(self) -> dict:
        return {
            "active_sessions": len(self.sessions),
            "total_messages": sum(
                s.message_count for s in self.sessions.values()
            ),
        }


# ═══════════════════════════════════════════════════════════════
# GATEWAY
# ═══════════════════════════════════════════════════════════════

class Gateway:
    def __init__(self, provider: LLMProvider):
        self.sessions = SessionManager()
        self.provider = provider
        self.connected_clients: dict[str, set] = {}
        self.boot_time = time.time()

    # ── message handling ──────────────────────────────────────

    async def handle_message(self, peer_id: str, data: dict) -> dict:
        msg_type = data.get("type", "message")

        if msg_type == "message":
            return await self._handle_chat(peer_id, data)
        elif msg_type == "command":
            return self._handle_command(peer_id, data)
        elif msg_type == "ping":
            return {"type": "pong", "time": time.time()}
        elif msg_type == "status":
            return self._get_status(peer_id)
        else:
            return {"type": "error", "text": f"Unknown type: {msg_type}"}

    async def _handle_chat(self, peer_id: str, data: dict) -> dict:
        text = data.get("text", "").strip()
        if not text:
            return {"type": "error", "text": "Empty message"}

        if text.startswith("/"):
            return self._handle_command(peer_id, {"command": text})

        session = self.sessions.get_or_create(peer_id)
        session.add("user", text)

        print(f"  [{peer_id}] User: {text[:60]}...")

        try:
            result = await self.provider.chat(
                messages=session.messages,
                system=SYSTEM_PROMPT,
            )

            session.add("assistant", result["text"])
            print(f"  [{peer_id}] Jarvis: {result['text'][:60]}...")

            return {
                "type": "response",
                "text": result["text"],
                "session_id": session.id,
                "tokens": {
                    "input": result["input_tokens"],
                    "output": result["output_tokens"],
                },
            }

        except Exception as e:
            print(f"  [!] LLM Error: {e}")
            return {"type": "error", "text": f"LLM error: {str(e)}"}

    def _handle_command(self, peer_id: str, data: dict) -> dict:
        cmd = data.get("command", "").strip().lower()

        if cmd in ("/new", "/reset"):
            session = self.sessions.get_or_create(peer_id)
            session.reset()
            return {"type": "system", "text": "Session reset. Fresh start, sir."}

        elif cmd == "/status":
            return self._get_status(peer_id)

        elif cmd == "/sessions":
            stats = self.sessions.get_stats()
            return {
                "type": "system",
                "text": (
                    f"Active sessions: {stats['active_sessions']}\n"
                    f"Total messages: {stats['total_messages']}"
                ),
            }

        elif cmd.startswith("/model"):
            parts = cmd.split(maxsplit=1)
            if len(parts) > 1:
                self.provider.model = parts[1]
                return {
                    "type": "system",
                    "text": f"Model switched to: {parts[1]}",
                }
            return {
                "type": "system",
                "text": f"Current model: {self.provider.model}\nUsage: /model <model-name>",
            }

        elif cmd == "/help":
            return {
                "type": "system",
                "text": (
                    "Available commands:\n"
                    "  /new or /reset — Reset conversation\n"
                    "  /status — Session + provider info\n"
                    "  /model <name> — Switch model\n"
                    "  /sessions — All active sessions\n"
                    "  /help — This message"
                ),
            }

        return {"type": "system", "text": f"Unknown command: {cmd}"}

    def _get_status(self, peer_id: str) -> dict:
        session = self.sessions.get_or_create(peer_id)
        uptime = int(time.time() - self.boot_time)
        return {
            "type": "status",
            "session_id": session.id,
            "messages": session.message_count,
            "history_size": len(session.messages),
            "model": f"{self.provider.name} / {self.provider.model}",
            "uptime_seconds": uptime,
            "connected_clients": sum(
                len(ws) for ws in self.connected_clients.values()
            ),
        }

    # ── WebSocket server ──────────────────────────────────────

    async def ws_handler(self, websocket):
        peer_id = None
        try:
            async for raw in websocket:
                data = json.loads(raw)

                if peer_id is None:
                    peer_id = data.get("peer_id", str(uuid.uuid4())[:8])
                    if peer_id not in self.connected_clients:
                        self.connected_clients[peer_id] = set()
                    self.connected_clients[peer_id].add(websocket)
                    print(f"  [*] Client connected: {peer_id}")

                    await websocket.send(json.dumps({
                        "type": "system",
                        "text": (
                            f"Jarvis online. Provider: {self.provider.name} "
                            f"({self.provider.model}). How can I help you, sir?"
                        ),
                    }))

                await websocket.send(json.dumps({"type": "typing"}))

                result = await self.handle_message(peer_id, data)
                await websocket.send(json.dumps(result))

        except Exception as e:
            print(f"  [!] WebSocket error: {e}")
        finally:
            if peer_id and peer_id in self.connected_clients:
                self.connected_clients[peer_id].discard(websocket)
                if not self.connected_clients[peer_id]:
                    del self.connected_clients[peer_id]
                print(f"  [-] Client disconnected: {peer_id}")

    # ── start ─────────────────────────────────────────────────

    async def start(self):
        import websockets

        print()
        print("=" * 52)
        print("  JARVIS GATEWAY")
        print(f"  WebSocket:  ws://{HOST}:{PORT}")
        print(f"  Provider:   {self.provider.name}")
        print(f"  Model:      {self.provider.model}")
        print("  Status:     ONLINE")
        print("=" * 52)
        print()
        print("  Open webchat.html in your browser to chat.")
        print("  Press Ctrl+C to stop.\n")

        async with websockets.serve(
            self.ws_handler, HOST, PORT,
            ping_interval=30,
            ping_timeout=10,
        ):
            await asyncio.Future()


# ═══════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # parse --provider flag
    forced = None
    if "--provider" in sys.argv:
        idx = sys.argv.index("--provider")
        if idx + 1 < len(sys.argv):
            forced = sys.argv[idx + 1]

    provider = detect_provider(forced)
    gateway = Gateway(provider)

    try:
        asyncio.run(gateway.start())
    except KeyboardInterrupt:
        print("\n  Jarvis signing off. Goodbye, sir.")