"""A2A Agent Identity — Runtime for agent cards, registry, streaming negotiation, and persistent identity.

Provides:
- AgentCard: load/validate Google A2A-draft agent cards from JSON
- AgentRegistry: in-memory registry with discovery, task negotiation, and stream handlers
- AgentIdentity: persistent Ed25519 identity for task signing and verification
- TaskHandle: async handle for in-flight A2A tasks with streaming chunk delivery
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional

# Ed25519 via cryptography (preferred) or nacl fallback
# We keep the dependency optional so pure-Python tests can still run.
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    _HAS_CRYPTO = True
except Exception:  # pragma: no cover
    _HAS_CRYPTO = False

logger = logging.getLogger(__name__)

__all__ = [
    "AgentCard",
    "AgentRegistry",
    "AgentIdentity",
    "TaskHandle",
    "TaskState",
    "NegotiationError",
    "ValidationError",
]


# ── exceptions ──────────────────────────────────────────────

class ValidationError(ValueError):
    """Raised when an agent card fails schema validation."""


class NegotiationError(RuntimeError):
    """Raised when task negotiation fails (network, auth, or capability mismatch)."""


# ── Task state ──────────────────────────────────────────────

class TaskState(Enum):
    """Lifecycle states for an in-flight A2A task."""
    PENDING = auto()
    SUBMITTED = auto()
    STREAMING = auto()
    COMPLETED = auto()
    FAILED = auto()


# ── AgentCard ───────────────────────────────────────────────

@dataclass(frozen=True)
class AgentCard:
    """Google A2A spec draft agent card.

    Frozen and slotted for hashability and memory efficiency.
    """

    name: str
    version: str
    description: str
    url: str
    capabilities: Dict[str, Any]
    skills: List[Dict[str, Any]]
    authentication: Dict[str, Any]
    default_input_content_type: str = "application/json"
    default_output_content_type: str = "application/json"

    # ── validation ────────────────────────────────────────

    # Class-level validation constants (excluded from dataclass fields)
    _REQUIRED_TOP_LEVEL: ClassVar[set] = {"name", "version", "description", "capabilities", "skills"}
    _REQUIRED_CAPABILITY_BOOLS: ClassVar[set] = {"streaming", "pushNotifications"}
    _REQUIRED_SKILL_FIELDS: ClassVar[set] = {"id", "name", "description", "tags", "examples"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCard":
        """Validate and construct from a parsed dict."""
        cls._validate_top_level(data)
        cls._validate_capabilities(data.get("capabilities", {}))
        cls._validate_skills(data.get("skills", []))
        cls._validate_authentication(data.get("authentication", {}))

        return cls(
            name=data["name"],
            version=data["version"],
            description=data.get("description", ""),
            url=data.get("url", ""),
            capabilities=data["capabilities"],
            skills=data["skills"],
            authentication=data.get("authentication", {}),
            default_input_content_type=data.get(
                "defaultInputContentType", "application/json"
            ),
            default_output_content_type=data.get(
                "defaultOutputContentType", "application/json"
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> "AgentCard":
        """Parse from a JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValidationError("Agent card JSON must be an object")
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentCard":
        """Load from a JSON file path."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    @classmethod
    def _validate_top_level(cls, data: Dict[str, Any]) -> None:
        missing = cls._REQUIRED_TOP_LEVEL - set(data.keys())
        if missing:
            raise ValidationError(f"Missing required top-level fields: {sorted(missing)}")

    @classmethod
    def _validate_capabilities(cls, caps: Dict[str, Any]) -> None:
        if not isinstance(caps, dict):
            raise ValidationError("'capabilities' must be an object")
        missing = cls._REQUIRED_CAPABILITY_BOOLS - set(caps.keys())
        if missing:
            raise ValidationError(f"Missing required capability fields: {sorted(missing)}")
        for key in cls._REQUIRED_CAPABILITY_BOOLS:
            if not isinstance(caps[key], bool):
                raise ValidationError(f"Capability '{key}' must be a boolean")

    @classmethod
    def _validate_skills(cls, skills: Any) -> None:
        if not isinstance(skills, list):
            raise ValidationError("'skills' must be an array")
        for idx, skill in enumerate(skills):
            if not isinstance(skill, dict):
                raise ValidationError(f"Skill at index {idx} must be an object")
            missing = cls._REQUIRED_SKILL_FIELDS - set(skill.keys())
            if missing:
                raise ValidationError(
                    f"Skill at index {idx} missing fields: {sorted(missing)}"
                )
            if not isinstance(skill.get("tags"), list):
                raise ValidationError(f"Skill at index {idx}: 'tags' must be an array")
            if not isinstance(skill.get("examples"), list):
                raise ValidationError(f"Skill at index {idx}: 'examples' must be an array")

    @classmethod
    def _validate_authentication(cls, auth: Dict[str, Any]) -> None:
        if not isinstance(auth, dict):
            raise ValidationError("'authentication' must be an object")
        schemes = auth.get("schemes")
        if schemes is not None and not isinstance(schemes, list):
            raise ValidationError("'authentication.schemes' must be an array")

    # ── helpers ─────────────────────────────────────────────

    def supports_skill(self, skill_id: str) -> bool:
        """True if this agent advertises a skill with the given id."""
        return any(s.get("id") == skill_id for s in self.skills)

    def capability(self, key: str, default: Any = None) -> Any:
        """Read a capability flag with a fallback default."""
        return self.capabilities.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to a plain dict."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "url": self.url,
            "capabilities": self.capabilities,
            "skills": self.skills,
            "authentication": self.authentication,
            "defaultInputContentType": self.default_input_content_type,
            "defaultOutputContentType": self.default_output_content_type,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ── TaskHandle ──────────────────────────────────────────────

@dataclass
class TaskHandle:
    """Async handle for an in-flight A2A task.

    Holds the task ID, target agent, current state, a queue of streamed
    response chunks, and an optional error.
    """

    task_id: str
    agent_id: str
    task_type: str
    payload: Dict[str, Any]
    state: TaskState = TaskState.PENDING
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    _chunk_event: asyncio.Event = field(default_factory=asyncio.Event)
    _loop: Optional[asyncio.AbstractEventLoop] = None

    def __post_init__(self) -> None:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
        self._chunk_event.clear()

    def set_state(self, state: TaskState) -> None:
        """Advance the task state machine."""
        self.state = state

    def deliver_chunk(self, chunk: Dict[str, Any]) -> None:
        """Append a streamed response chunk and signal waiters."""
        self.chunks.append(chunk)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._chunk_event.set)

    async def next_chunk(self, timeout: float | None = None) -> Optional[Dict[str, Any]]:
        """Wait for and return the next chunk, or None on timeout."""
        if not self.chunks:
            try:
                await asyncio.wait_for(self._chunk_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
        self._chunk_event.clear()
        return self.chunks[-1] if self.chunks else None


# ── AgentRegistry ───────────────────────────────────────────

class AgentRegistry:
    """In-memory registry of known agents (local + remote).

    Provides:
    - Agent discovery via nexus query
    - Task negotiation with streaming response support
    - Per-agent stream handler registration
    """

    def __init__(self, identity: Optional[AgentIdentity] = None) -> None:
        self._agents: Dict[str, AgentCard] = {}
        self._stream_handlers: Dict[str, Callable[[str, Dict[str, Any]], None]] = {}
        self._pending_tasks: Dict[str, TaskHandle] = {}
        self._identity = identity
        self._fallback_local_cards: Dict[str, AgentCard] = {}

    # ── registration ──────────────────────────────────────

    def register(self, agent_id: str, card: AgentCard) -> None:
        """Register or update an agent card."""
        self._agents[agent_id] = card
        logger.debug("Registered agent %s (%s)", agent_id, card.name)

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the registry."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    def get(self, agent_id: str) -> Optional[AgentCard]:
        """Return the card for a known agent, or None."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[str]:
        """Return a list of known agent IDs."""
        return list(self._agents.keys())

    def load_local_cards(self, directory: str | Path) -> int:
        """Load all JSON agent cards from a directory."""
        d = Path(directory)
        count = 0
        for path in d.glob("*.json"):
            try:
                card = AgentCard.from_file(path)
                agent_id = path.stem
                self.register(agent_id, card)
                self._fallback_local_cards[agent_id] = card
                count += 1
            except ValidationError as exc:
                logger.warning("Skipping invalid card %s: %s", path, exc)
        return count

    # ── discovery ───────────────────────────────────────────

    async def discover(self, nexus_url: str) -> List[AgentCard]:
        """Query the nexus for available agents and register them.

        Falls back to locally cached cards if the nexus is unreachable.
        """
        url = nexus_url.rstrip("/") + "/.well-known/agent-cards"
        try:
            cards = await self._http_get_agent_cards(url)
            for agent_id, card in cards:
                self.register(agent_id, card)
            return [c for _, c in cards]
        except NegotiationError as exc:
            logger.warning("Nexus discovery failed (%s); falling back to local cards", exc)
            # Fallback: return any previously loaded local cards
            fallback = list(self._fallback_local_cards.values())
            if fallback:
                logger.info("Using %d local fallback cards", len(fallback))
                return fallback
            raise

    async def _http_get_agent_cards(self, url: str) -> List[tuple[str, AgentCard]]:
        """HTTP GET agent card list.

        In a real deployment this would use aiohttp/httpx.
        Here we provide an async hook that tests may patch.
        """
        # Placeholder: subclasses or monkey-patching in tests should override
        raise NegotiationError("No HTTP client configured; override _http_get_agent_cards")

    # ── task negotiation ────────────────────────────────────

    def negotiate_task(
        self,
        agent_id: str,
        task_type: str,
        payload: Dict[str, Any],
    ) -> TaskHandle:
        """Send a task to an agent and return a handle.

        Does not actually perform network I/O here; the caller (or an
        async loop) drives the handle to SUBMITTED/STREAMING by calling
        _submit_task_handle() with a transport-specific submit coroutine.
        """
        card = self.get(agent_id)
        if card is None:
            raise NegotiationError(f"Unknown agent: {agent_id}")
        if not card.supports_skill(task_type):
            raise NegotiationError(
                f"Agent {agent_id} does not support skill '{task_type}'"
            )
        if not card.capability("streaming", False):
            logger.warning(
                "Agent %s does not advertise streaming support for task %s",
                agent_id,
                task_type,
            )

        handle = TaskHandle(
            task_id=str(uuid.uuid4()),
            agent_id=agent_id,
            task_type=task_type,
            payload=payload,
        )
        self._pending_tasks[handle.task_id] = handle
        handle.set_state(TaskState.PENDING)
        logger.debug("Created task handle %s for %s:%s", handle.task_id, agent_id, task_type)
        return handle

    async def submit_task(
        self,
        handle: TaskHandle,
        submit_fn: Callable[[TaskHandle], Any],
    ) -> TaskHandle:
        """Drive a handle through submission using an async transport callable.

        ``submit_fn`` receives the handle and is expected to deliver chunks
        via ``handle.deliver_chunk()`` and set state to COMPLETED or FAILED.
        """
        handle.set_state(TaskState.SUBMITTED)
        try:
            await submit_fn(handle)
        except Exception as exc:
            handle.set_state(TaskState.FAILED)
            handle.error = str(exc)
            logger.exception("Task %s submission failed", handle.task_id)
        return handle

    # ── stream handlers ─────────────────────────────────────

    def register_stream_handler(
        self,
        agent_id: str,
        callback: Callable[[str, Dict[str, Any]], None],
    ) -> None:
        """Register a callback invoked on every streamed chunk from *agent_id*.

        Signature: callback(agent_id: str, chunk: dict) -> None
        """
        self._stream_handlers[agent_id] = callback
        logger.debug("Registered stream handler for agent %s", agent_id)

    def unregister_stream_handler(self, agent_id: str) -> bool:
        """Remove the stream handler for an agent."""
        return self._stream_handlers.pop(agent_id, None) is not None

    def dispatch_chunk(self, agent_id: str, chunk: Dict[str, Any]) -> None:
        """Dispatch a chunk to the registered handler for *agent_id*, if any."""
        handler = self._stream_handlers.get(agent_id)
        if handler is not None:
            try:
                handler(agent_id, chunk)
            except Exception:
                logger.exception("Stream handler for %s raised an exception", agent_id)
        # Also deliver to any pending task handle matching this agent
        for handle in self._pending_tasks.values():
            if handle.agent_id == agent_id and handle.state in (
                TaskState.SUBMITTED,
                TaskState.STREAMING,
            ):
                handle.deliver_chunk(chunk)
                if handle.state == TaskState.SUBMITTED:
                    handle.set_state(TaskState.STREAMING)

    # ── cleanup ─────────────────────────────────────────────

    def close_task(self, task_id: str) -> Optional[TaskHandle]:
        """Remove a task from the pending map and return it."""
        return self._pending_tasks.pop(task_id, None)


# ── AgentIdentity ───────────────────────────────────────────

class AgentIdentity:
    """Persistent Ed25519 identity for an A2A agent.

    Signs outgoing tasks so that downstream agents (and SignedWAL)
    can verify provenance. Also verifies signatures on incoming tasks
    from remote agents.

    Key material is stored under ``base_dir / "agent_id.pem"`` (private)
    and ``base_dir / "agent_id.pub"`` (public).
    """

    def __init__(
        self,
        agent_id: str,
        base_dir: str | Path = ".a2a/keys",
    ) -> None:
        self.agent_id = agent_id
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._private_key: Any = None
        self._public_key: Any = None
        self._load_or_generate()

    # ── key management ────────────────────────────────────

    def _private_path(self) -> Path:
        return self.base_dir / f"{self.agent_id}.pem"

    def _public_path(self) -> Path:
        return self.base_dir / f"{self.agent_id}.pub"

    def _load_or_generate(self) -> None:
        pem_path = self._private_path()
        pub_path = self._public_path()
        if pem_path.exists() and pub_path.exists():
            self._load_keys(pem_path, pub_path)
        else:
            if _HAS_CRYPTO:
                self._generate_keys(pem_path, pub_path)
            else:
                logger.warning(
                    "cryptography not available; identity will use no-op signing"
                )

    def _load_keys(self, pem_path: Path, pub_path: Path) -> None:
        if not _HAS_CRYPTO:
            return
        with open(pem_path, "rb") as fh:
            self._private_key = serialization.load_pem_private_key(fh.read(), password=None)
        with open(pub_path, "rb") as fh:
            self._public_key = serialization.load_pem_public_key(fh.read())

    def _generate_keys(self, pem_path: Path, pub_path: Path) -> None:
        if not _HAS_CRYPTO:
            return
        self._private_key = Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        pem = self._private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        pub = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        with open(pem_path, "wb") as fh:
            fh.write(pem)
        os.chmod(pem_path, 0o600)
        with open(pub_path, "wb") as fh:
            fh.write(pub)
        logger.info("Generated new Ed25519 keypair for %s", self.agent_id)

    # ── signing / verification ──────────────────────────────

    def sign_task(self, payload: Dict[str, Any]) -> str:
        """Sign a task payload dict and return a base64-encoded signature.

        The payload is canonicalised (sorted keys, no whitespace) before
        signing to ensure deterministic signatures.
        """
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        message = canonical.encode("utf-8")
        if _HAS_CRYPTO and self._private_key is not None:
            sig = self._private_key.sign(message)
            import base64
            return base64.b64encode(sig).decode("ascii")
        # Fallback no-op signature when cryptography is unavailable
        import hashlib
        return hashlib.sha256(message).hexdigest()[:64]

    def verify_task(self, payload: Dict[str, Any], signature: str, public_key_pem: str | None = None) -> bool:
        """Verify a task signature.

        If ``public_key_pem`` is provided, the signature is checked against
        that key. Otherwise the local public key is used (self-verification,
        mainly for tests).
        """
        if not _HAS_CRYPTO:
            # No-op fallback: just validate the hash-like string length
            return len(signature) == 64
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        message = canonical.encode("utf-8")
        import base64
        try:
            sig_bytes = base64.b64decode(signature)
        except Exception:
            return False
        key = self._public_key
        if public_key_pem is not None:
            key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
        if key is None:
            return False
        try:
            key.verify(sig_bytes, message)
            return True
        except InvalidSignature:
            return False

    def public_key_pem(self) -> str:
        """Return the public key as a PEM string for sharing with peers."""
        if not _HAS_CRYPTO or self._public_key is None:
            return ""
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

    @property
    def has_crypto(self) -> bool:
        return _HAS_CRYPTO and self._private_key is not None
