"""
Inference script and Baseline Agent for OpenEnv Moderation Task.
Provides evaluation and deterministic testing functionality.
"""
import os
import sys

# Append parent dir to allow direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging


from llm_client import call_llm_api

from env.environment import ModerationEnv
from env.models import ActionModel
from env.tasks import TASKS
from env.grader import grade_trajectory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

class BaselineAgent:
    def __init__(self):
        self.model = MODEL_NAME
        
    def _call_llm(self, prompt: str) -> dict:
        return call_llm_api(prompt, self.model)

    def solve_task(self, env: ModerationEnv) -> float:
        obs = env.reset()
        done = False
        trajectory = []
        rewards = []
        step_count = 0
        
        log_start(task=env.task.task_id, env="OpenEnv-Moderation", model=self.model)
        
        system_prompt = """
        You are an intelligent moderation agent. At each step, analyze the current observation and tool results.
        If you need more information, return a JSON requesting a tool:
        {"action_type": "TOOL", "next_tool": "<tool_name>", "justification": "<reason>"}
        Allowed tools: extract_claims, verify_source, run_fact_check, update_risk_score
        
        If you have enough information, make a moderation decision:
        {"action_type": "MODERATE", "classification": "REAL|MISLEADING|FAKE|HARMFUL", "confidence": <float 0-1>, "action": "ALLOW|FLAG|REMOVE|ESCALATE", "justification": "<reason>"}
        """
        
        prompt_history = system_prompt + "\n\n"
        final_decision = None
        
        while not done:
            obs_dict = obs.model_dump()
            prompt = prompt_history + f"\nObservation: {json.dumps(obs_dict)}\n"
            
            action_dict = self._call_llm(prompt)
            # Make sure keys match Pydantic model
            if action_dict.get("action_type") == "TOOL":
                action = ActionModel(
                    action_type="TOOL",
                    next_tool=action_dict["next_tool"],
                    justification=action_dict.get("justification", "")
                )
            else:
                action = ActionModel(
                    action_type="MODERATE",
                    classification=action_dict["classification"],
                    confidence=action_dict["confidence"],
                    action=action_dict["action"],
                    justification=action_dict.get("justification", "")
                )
                
            obs, reward, done, info = env.step(action)
            step_count += 1
            step_reward_val = reward.step_reward
            rewards.append(step_reward_val)
            
            # Record trajectory
            trajectory.append(action_dict)
            prompt_history += f"\nObservation: {json.dumps(obs_dict)}\nAction Taken: {json.dumps(action_dict)}\n"
            
            action_str = action_dict.get('next_tool') if action.action_type == 'TOOL' else action_dict.get('action')
            log_step(step=step_count, action=str(action_str), reward=step_reward_val, done=done, error=None)
            
            if action.action_type == "MODERATE":
                final_decision = action_dict
                
        # If max steps reached without decision, make a random/default one for grader
        if final_decision is None:
            final_decision = {
                "action_type": "MODERATE",
                "classification": "UNKNOWN",
                "confidence": 0.0,
                "action": "ALLOW",
            }
                
        # Calculate grade
        score = grade_trajectory(trajectory, final_decision, env.task.task_id)
        success = score > 0.5
        log_end(success=success, steps=step_count, score=score, rewards=rewards)
        return score

if __name__ == "__main__":
    agent = BaselineAgent()
    
    total_score = 0
    scores = {}
    
    for t in TASKS:
        env = ModerationEnv(task_id=t.task_id)
        logger.info(f"Running task: {t.task_id} ({t.difficulty})")
        
        score = agent.solve_task(env)
        scores[t.task_id] = score
        total_score += score
        
        logger.info(f"Score for {t.task_id}: {score:.2f}")
        
    avg_score = total_score / len(TASKS)
    logger.info(f"Average Score across all tasks: {avg_score:.2f}")
    
    print(json.dumps({
        "per_task": scores,
        "average_score": avg_score
    }, indent=2))
