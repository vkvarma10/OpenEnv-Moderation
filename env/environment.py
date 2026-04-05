"""
Core definition of the OpenEnv Moderation Environment setup.
Implements the reset, step, and state logic for simulating moderation tasks.
"""
from env.models import ObservationModel, ActionModel, RewardModel, StateModel, TaskModel
from env.tasks import get_task
from env.reward import calculate_reward
from env import tools

class ModerationEnv:
    """
    Simulates a moderation pipeline platform where an agent evaluates content.
    Follows OpenEnv API definitions.
    """
    def __init__(self, task_id: str):
        self.task: TaskModel = get_task(task_id)
        self.max_steps = 10
        self._state = None
        
    def reset(self) -> ObservationModel:
        self._state = StateModel(
            content_id=self.task.task_id,
            content_text=self.task.content,
            source_type=self.task.source_type,
            source_reliability_score=0.5, # initial unknown
            user_reputation_score=0.5, # initial unknown
            detected_claims=[],
            fact_check_results={},
            risk_score=0.5, # initial neutral
            uncertainty_level=1.0, # high uncertainty at start
            action_history=[],
            step_count=0,
            ground_truth_label=self.task.ground_truth_label,
            ground_truth_action=self.task.ground_truth_action
        )
        return self._get_observation()

    def step(self, action: ActionModel):
        if self._state is None:
            raise RuntimeError("Environment must be reset before calling step.")
            
        self._state.step_count += 1
        self._state.action_history.append(action)
        
        tool_results = {}
        done = False
        info = {}
        
        # State Evolution
        if action.action_type == "TOOL":
            self._state.uncertainty_level = max(0.0, self._state.uncertainty_level - 0.2)
            
            if action.next_tool == "extract_claims":
                claims = tools.extract_claims(self._state.content_text)
                self._state.detected_claims = claims
                tool_results["extracted_claims"] = claims
                
            elif action.next_tool == "verify_source":
                score = tools.verify_source(self._state.source_type, self.task.metadata)
                self._state.source_reliability_score = score
                tool_results["source_reliability"] = score
                
            elif action.next_tool == "run_fact_check":
                if not self._state.detected_claims:
                    tool_results["fact_check"] = {"error": "No claims extracted yet. Run extract_claims first."}
                else:
                    results = tools.run_fact_check(self._state.detected_claims)
                    self._state.fact_check_results = results
                    tool_results["fact_check"] = results
                    
            elif action.next_tool == "update_risk_score":
                risk = tools.calculate_risk_score(
                    self._state.detected_claims,
                    self._state.source_reliability_score,
                    self._state.content_text
                )
                self._state.risk_score = risk
                tool_results["risk_score"] = risk
                
        elif action.action_type == "MODERATE":
            done = True
            
        if self._state.step_count >= self.max_steps:
            done = True
            info["reason"] = "max_steps_reached"
            
        # Calculate Reward
        reward_model = calculate_reward(action, self._state, done)
        
        # Build Observation
        obs = self._get_observation(tool_results)
        
        info["step_reward_components"] = reward_model.components
        
        return obs, reward_model, done, info

    def state(self) -> StateModel:
        if self._state is None:
            raise RuntimeError("Environment must be reset before getting state.")
        return self._state

    def _get_observation(self, tool_results: dict = None) -> ObservationModel:
        if tool_results is None:
            tool_results = {}
            
        return ObservationModel(
            content_text=self._state.content_text,
            metadata={"source_type": self._state.source_type},  # Partial metadata
            previous_actions=[a.dict() for a in self._state.action_history],
            tool_results=tool_results,
            current_step=self._state.step_count
        )
