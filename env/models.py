"""
Pydantic data models for the OpenEnv Moderation Environment state, observation, and actions.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ObservationModel(BaseModel):
    content_text: str = Field(..., description="The text content to be moderated.")
    metadata: dict = Field(default_factory=dict, description="Limited metadata about the source or user.")
    previous_actions: List[dict] = Field(default_factory=list, description="Actions previously taken by the agent.")
    tool_results: dict = Field(default_factory=dict, description="Results from recently executed tools.")
    current_step: int = Field(..., description="Current step in the moderation process.")

class ActionModel(BaseModel):
    action_type: Literal["MODERATE", "TOOL"] = Field(..., description="Whether this is a final moderation action or a tool use.")
    classification: Optional[Literal["REAL", "MISLEADING", "FAKE", "HARMFUL", "UNKNOWN"]] = Field(None, description="Current classification assessment.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the current assessment.")
    action: Optional[Literal["ALLOW", "FLAG", "REMOVE", "ESCALATE"]] = Field(None, description="Final moderation action (if MODERATE).")
    justification: str = Field(..., description="Reasoning for this step's action.")
    next_tool: Optional[Literal["extract_claims", "verify_source", "run_fact_check", "update_risk_score"]] = Field(None, description="Tool to use (if TOOL).")

class RewardModel(BaseModel):
    step_reward: float = Field(..., description="Reward for the current step.")
    components: dict = Field(default_factory=dict, description="Breakdown of reward sources.")
    is_terminal: bool = Field(False, description="Whether the episode is concluded.")

class StateModel(BaseModel):
    content_id: str
    content_text: str
    source_type: str
    source_reliability_score: float
    user_reputation_score: float
    detected_claims: List[str]
    fact_check_results: dict
    risk_score: float
    uncertainty_level: float
    action_history: List[ActionModel]
    step_count: int
    ground_truth_label: str
    ground_truth_action: str

class TaskModel(BaseModel):
    task_id: str
    difficulty: Literal["EASY", "MEDIUM", "HARD"]
    content: str
    source_type: str
    source_reliability: float
    user_reputation: float
    metadata: dict
    ground_truth_label: Literal["REAL", "MISLEADING", "FAKE", "HARMFUL"]
    ground_truth_action: Literal["ALLOW", "FLAG", "REMOVE", "ESCALATE"]
    ground_truth_claims: List[str]
