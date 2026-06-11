import asyncio
import threading

MAX_HISTORY = 2000


class EventBus:
    """Thread-safe event bus.

    Worker threads call publish() with a JSON-serializable dict; the bus fans
    the event out to every connected WebSocket client on the asyncio loop.
    A bounded history buffer is replayed to clients that connect late (e.g.
    a page refresh keeps the conversation on screen).
    """

    def __init__(self):
        self._loop = None
        self._clients = set()
        self._lock = threading.Lock()
        self._history = []

    def set_loop(self, loop):
        self._loop = loop

    async def register(self, ws):
        with self._lock:
            self._clients.add(ws)
            history = list(self._history)
        for event in history:
            await ws.send_json(event)

    def unregister(self, ws):
        with self._lock:
            self._clients.discard(ws)

    def publish(self, event: dict):
        """Safe to call from any thread."""
        if event.get("replay", True):
            with self._lock:
                self._history.append(event)
                if len(self._history) > MAX_HISTORY:
                    self._history = self._history[-MAX_HISTORY:]
        with self._lock:
            clients = list(self._clients)
        if self._loop is None:
            return
        for ws in clients:
            asyncio.run_coroutine_threadsafe(self._safe_send(ws, event), self._loop)

    async def _safe_send(self, ws, event):
        try:
            await ws.send_json(event)
        except Exception:
            self.unregister(ws)


bus = EventBus()
