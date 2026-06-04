import json


class WSManager:
    def __init__(self):
        self.connections = set()

    async def connect(self, ws):
        self.connections.add(ws)

    async def disconnect(self, ws):
        self.connections.discard(ws)

    async def broadcast(self, data):
        payload = json.dumps(data)
        dead = set()
        for ws in self.connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self.connections -= dead

    @property
    def count(self):
        return len(self.connections)
