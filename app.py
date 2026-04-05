"""
FastAPI application for OpenEnv Moderation Environment.
Exposes standard endpoints (/reset, /step, /state) required for OpenEnv validation.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys

# Ensure local imports work correctly regardless of how the script is executed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import ModerationEnv
from env.models import ActionModel

app = FastAPI(
    title="OpenEnv Moderation API",
    description="API for evaluating moderation agents.",
    version="1.0.0"
)

# Simple in-memory storage for active environments keyed by session id
envs = {}

class ResetRequest(BaseModel):
    session_id: str = "default_session"
    task_id: str = "task_easy_01"

class StepRequest(BaseModel):
    session_id: str
    action: ActionModel

@app.get("/")
def health_check():
    """Root endpoint to check if the server is running."""
    return {"status": "ok"}

@app.post("/reset")
def reset_env(req: ResetRequest):
    """
    Initialize or reset the environment correctly.
    Returns the initial observation space.
    """
    try:
        env = ModerationEnv(task_id=req.task_id)
        obs = env.reset()
        envs[req.session_id] = env
        return {"observation": obs.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(req: StepRequest):
    """
    Take an action in the environment.
    Returns observation, reward, done status, and info.
    """
    if req.session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    
    env = envs[req.session_id]
    try:
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state/{session_id}")
def get_state(session_id: str):
    """Retrieve the full internal state of an active session."""
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"state": envs[session_id].state().model_dump()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
