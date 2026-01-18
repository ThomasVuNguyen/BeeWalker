"""
BeeWalker Web Dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from pathlib import Path
from typing import List, Dict
import json

app = FastAPI()

# Fix path to static files to be absolute
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        # Convert to JSON string
        txt = json.dumps(message)
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(txt)
            except:
                to_remove.append(connection)
        
        for conn in to_remove:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# Global training state
class TrainingState:
    generation: int = 0
    mean_reward: float = 0.0
    status: str = "Idle"
    history: list = []
    viz_data: dict = {}

state = TrainingState()

@app.get("/")
async def get():
    return HTMLResponse(open(STATIC_DIR / "index.html").read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send current state on connect
        await websocket.send_text(json.dumps({
            "type": "update",
            "data": {
                "generation": state.generation,
                "mean_reward": state.mean_reward,
                "status": state.status
            }
        }))
        while True:
            # Keep alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except:
        manager.disconnect(websocket)

@app.post("/update")
async def update_state(data: dict):
    state.generation = data.get("generation", 0)
    state.mean_reward = data.get("mean_reward", 0.0)
    state.status = data.get("status", "Idle")
    
    # Keep history for new clients
    state.history.append({
        "generation": state.generation,
        "mean_reward": state.mean_reward
    })
    
    # Broadcast to all connected clients
    await manager.broadcast({
        "type": "update",
        "data": {
            "generation": state.generation,
            "mean_reward": state.mean_reward,
            "status": state.status
        }
    })
    return {"status": "ok"}

@app.post("/viz")
async def update_viz(data: dict):
    state.viz_data = data
    # Broadcast visualization data (now contains frames array)
    await manager.broadcast({
        "type": "viz",
        "data": data
    })
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1306)
