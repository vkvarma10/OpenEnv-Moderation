# AI-Powered Misinformation Moderation Environment

## Motivation
With the unprecedented scale of user-generated content, large platforms are increasingly struggling to combat misinformation during critical events like elections or pandemics. This system provides a **production-ready OpenEnv environment** to train and evaluate AI agents on multi-step moderation logic under uncertainty—simulating the actual flow of a Trust & Safety team.

## System Architecture

```text
+-------------------+      (Actions)       +---------------------+
|                   | -------------------> |                     |
|  AI Agent (LLM)   |                      |  ModerationEnv      |
|                   | <------------------- |  (State & Reward)   |
+-------------------+    (Obs & Reward)    +---------------------+
        |                                            |
        v                                            v
+-------------------+                      +---------------------+
|                   |                      |   Tooling & Mock    |
|   OpenAI API      |                      |   Intelligence      |
|                   |                      |   Layer (BERT)      |
+-------------------+                      +---------------------+
```

## Action & Observation Design

### State (Hidden)
The state represents the full truth and metadata behind the content, including `risk_score`, `uncertainty_level`, `ground_truth_label`, and `detected_claims`. 

### Observation (Partial)
The agent is only given partial information (a Pydantic `ObservationModel`) containing `content_text`, limited metadata (e.g. `source_type`), tools execution history, and past actions.

### Action Space
Defined by `ActionModel`. The agent outputs either:
- **TOOL Use**: Requesting `extract_claims`, `verify_source`, `run_fact_check`, or `update_risk_score`.
- **MODERATE Decision**: The final classification (`REAL`, `MISLEADING`, `FAKE`, `HARMFUL`) and action (`ALLOW`, `FLAG`, `REMOVE`, `ESCALATE`) along with confidence and justification.

## Reward Strategy (Dense & Continuous)
The reward function evaluates every single step taken by the agent:
- **Positive Signals**: Meaningful reasoning (+0.1), correct classification (+0.4) and actionable moderation (+0.3), properly calibrated high confidence (+0.1).
- **Penalties**: False negatives/missing harmful content (-0.5), over-censorship (-0.3), uncalibrated high-confidence errors (-0.2), and repetitive tool loops (-0.3).

## Task Definitions
To assess agent reasoning, tasks dynamically stretch from basic text checking to multi-variable ambiguity:
- **EASY**: Clear distinction with high source reliability.
- **MEDIUM**: Misleading elements intertwined with part truth, requiring intermediate source investigations.
- **HARD**: Subtle or borderline misinformation mixing truth with complex nuances requiring multi-step investigation.

## Setup Instructions

1. **Clone the repository.**
2. **Install requirements:** `pip install -r requirements.txt`
3. **Execute baselines:** Ensure `PYTHONPATH` includes the base directory.
   ```bash
   export PYTHONPATH=$(pwd)
   python baseline/run_agent.py
   ```

## Example Interaction
```json
{
  "action_type": "TOOL",
  "next_tool": "extract_claims",
  "justification": "Before I can evaluate this article on pyramid construction, I need to know standard claims."
}
```

## Baseline Results
The mock baseline deterministic agent achieves:
1. `task_easy_01`: 0.8 / 1.0 (Efficiency caps out at certain heuristic points)
2. `task_medium_01`: 0.9 / 1.0 
3. `task_hard_01`: 0.7 / 1.0 (Struggles with deep subtlety without real LLM logic)
*These outputs depend on whether the actual OpenAI API resolves the trajectory or defaults to deterministic mocking.*
