"""A2A HTTP Server — lightweight, stdlib-only, thread-safe."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


class A2AServer:
    """Lightweight HTTP server for A2A Agent Cards and task dispatch.

    Uses only Python stdlib: http.server, socketserver, json, threading.
    """

    def __init__(self, port=8080, agents=None):
        self.port = port
        self.agents = dict(agents) if agents else {}
        self._lock = threading.Lock()
        self._server = None
        self._thread = None
        self._shutdown_event = threading.Event()
        # Base directory for resolving .well-known agent card files
        self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def register_agent(self, card_name, handler_func):
        """Register an agent handler.

        Args:
            card_name: Name used in the agent card filename (e.g. 'metronome').
            handler_func: Callable that receives a JSON task payload dict
                and returns a JSON-serializable response dict.
        """
        with self._lock:
            self.agents[card_name] = handler_func

    def _make_handler(self):
        """Factory that returns a request handler class bound to this server."""
        server = self

        class _Handler(BaseHTTPRequestHandler):
            """Request handler for A2A endpoints."""

            # Silence default logging
            def log_message(self, format, *args):
                pass

            def _send_json(self, status_code, body_dict):
                payload = json.dumps(body_dict, indent=2).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                """Serve static agent cards and health check."""
                if self.path == "/health":
                    self._send_json(200, {
                        "status": "ok",
                        "service": "a2a-server",
                        "agents": len(server.agents),
                    })
                    return
                if not self.path.startswith("/.well-known/agent-"):
                    self._send_json(404, {"error": "Not found"})
                    return

                # Extract card name from path
                # Path format: /.well-known/agent-{name}.json
                filename = self.path.split("/")[-1]
                if not filename.startswith("agent-") or not filename.endswith(".json"):
                    self._send_json(404, {"error": "Not found"})
                    return

                card_name = filename[len("agent-"): -len(".json")]

                with server._lock:
                    if card_name not in server.agents:
                        self._send_json(404, {"error": f"Unknown agent card: {card_name}"})
                        return

                # Load the agent card from .well-known directory
                card_path = os.path.join(server._base_dir, ".well-known", filename)
                if not os.path.exists(card_path):
                    self._send_json(404, {"error": f"Agent card file not found: {filename}"})
                    return

                try:
                    with open(card_path, "r", encoding="utf-8") as f:
                        card_data = json.load(f)
                except (OSError, json.JSONDecodeError) as exc:
                    self._send_json(500, {"error": f"Failed to read agent card: {exc}"})
                    return

                payload = json.dumps(card_data, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                """Dispatch A2A tasks at /a2a/tasks/send"""
                if self.path != "/a2a/tasks/send":
                    self._send_json(404, {"error": "Not found"})
                    return

                # Read body
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self._send_json(400, {"error": "Empty request body"})
                    return

                body = self.rfile.read(content_length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                    return

                # Dispatch based on agent name in payload
                # Per A2A spec, payload contains an agent or service identifier.
                # We support two patterns:
                #   1. { "agent": "metronome", "type": "tick", ... }
                #   2. { "type": "tick", ... } with agent inferred from type
                agent_name = payload.get("agent")
                if agent_name is None:
                    # Fallback: try to infer from the 'type' field mapping
                    task_type = payload.get("type", "")
                    type_to_agent = {
                        "tick": "metronome",
                        "set_bpm": "metronome",
                        "sync": "metronome",
                        "get_status": "metronome",
                        "queue_breed": "breeder",
                        "get_state": "breeder",
                        "get_stats": "breeder",
                        "emergency_stop": "breeder",
                        "get_activity": "grid",
                        "get_room_state": "grid",
                        "rebirth_room": "grid",
                        "check_constraints": "flux",
                        "get_violations": "flux",
                        "apply_feedback": "flux",
                    }
                    agent_name = type_to_agent.get(task_type)

                if agent_name is None:
                    self._send_json(400, {
                        "error": "Missing 'agent' field and unable to infer from 'type'"
                    })
                    return

                with server._lock:
                    handler = server.agents.get(agent_name)

                if handler is None:
                    self._send_json(404, {"error": f"No handler registered for agent: {agent_name}"})
                    return

                try:
                    result = handler(payload)
                except Exception as exc:
                    self._send_json(500, {"error": f"Handler error: {exc}"})
                    return

                self._send_json(200, result)

        return _Handler

    def start(self):
        """Run the HTTP server in a background daemon thread."""
        handler_class = self._make_handler()
        self._server = ThreadingHTTPServer(("", self.port), handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        """Shut down the server and join the background thread."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def url(self):
        """Return the base URL of the running server."""
        return f"http://localhost:{self.port}"


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server."""
    allow_reuse_address = True
    daemon_threads = True
