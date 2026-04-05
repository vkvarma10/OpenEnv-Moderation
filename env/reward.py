from env.models import ActionModel, StateModel, RewardModel

def calculate_reward(action: ActionModel, state: StateModel, is_terminal: bool) -> RewardModel:
    reward = 0.0
    components = {}
    
    gt_label = state.ground_truth_label
    gt_action = state.ground_truth_action
    
    if action.action_type == "TOOL":
        # Check redundancy
        if action.next_tool in [a.next_tool for a in state.action_history if a.action_type == "TOOL"]:
            comp_r = -0.3
            components["redundant_action"] = comp_r
            reward += comp_r
        else:
            comp_r = 0.1
            components["meaningful_reasoning"] = comp_r
            reward += comp_r
            
    elif action.action_type == "MODERATE":
        # Premature finalization? Did they use any tools?
        if len(state.action_history) == 0:
            comp_r = -0.2
            components["premature_finalization"] = comp_r
            reward += comp_r
            
        # Classification correctness
        if action.classification == gt_label:
            comp_r = 0.4
            components["classification_correctness"] = comp_r
            reward += comp_r
            
        # Moderation action correct?
        if action.action == gt_action:
            comp_r = 0.3
            components["correct_moderation_action"] = comp_r
            reward += comp_r
            
            # Correct escalation? (If action was ESCALATE and correct)
            if action.action == "ESCALATE":
                comp_r = 0.1
                components["correct_escalation"] = comp_r
                reward += comp_r
        else:
            # False negative (Missed harmful content)
            if gt_label in ["HARMFUL", "FAKE"] and action.action == "ALLOW":
                comp_r = -0.5
                components["false_negative"] = comp_r
                reward += comp_r
            # Over-censorship
            elif gt_label == "REAL" and action.action in ["FLAG", "REMOVE"]:
                comp_r = -0.3
                components["over_censorship"] = comp_r
                reward += comp_r
                
        # Confidence calibration
        confidence = action.confidence if action.confidence is not None else 0.5
        is_correct = (action.classification == gt_label) and (action.action == gt_action)
        
        if is_correct:
            if confidence > 0.8:
                comp_r = 0.1
                components["calibrated_confidence"] = comp_r
                reward += comp_r
        else:
            if confidence > 0.8:
                comp_r = -0.2
                components["high_confidence_wrong"] = comp_r
                reward += comp_r
            
    # Ensure reward is somewhat bounded/continuous per step.
    # Clip between -1.0 and 1.0 step-wise, but we just return sum.
    # Total episode reward should be accumulated by env.
    
    # Optionally add small step penalty to encourage efficiency
    reward -= 0.05
    components["step_penalty"] = -0.05

    return RewardModel(
        step_reward=reward,
        components=components,
        is_terminal=is_terminal
    )
