def grade_trajectory(trajectory: list, final_decision: dict, task_id: str) -> float:
    # A standalone grader computing a deterministic score [0.0, 1.0]
    from env.tasks import get_task
    task = get_task(task_id)
    
    score = 0.0
    
    gt_label = task.ground_truth_label
    gt_action = task.ground_truth_action
    
    # 1. Classification accuracy (0.3)
    if final_decision.get("classification") == gt_label:
        score += 0.3
        
    # 2. Action correctness (0.3)
    if final_decision.get("action") == gt_action:
        score += 0.3
        
    # 3. Efficiency (0.2)
    # Penalize too many steps, but also penalize 0 tools.
    num_tools = sum(1 for step in trajectory if step["action_type"] == "TOOL")
    if 1 <= num_tools <= 3:
        score += 0.2
    elif num_tools > 3:
        score += 0.1 # over-analysis
        
    # 4. Calibration (0.1)
    # Should have high confidence if correct, low if incorrect
    conf = final_decision.get("confidence", 0.5)
    is_correct = (final_decision.get("action") == gt_action)
    if is_correct and conf >= 0.8:
        score += 0.1
    elif not is_correct and conf < 0.5:
        score += 0.1
        
    # 5. Reasoning Quality (0.1)
    # Check if necessary tools for the task difficulty were used
    used_tools = [step["next_tool"] for step in trajectory if step["action_type"] == "TOOL"]
    
    if task.difficulty == "HARD" and "run_fact_check" in used_tools and "verify_source" in used_tools:
        score += 0.1
    elif task.difficulty == "MEDIUM" and ("run_fact_check" in used_tools or "verify_source" in used_tools):
        score += 0.1
    elif task.difficulty == "EASY":
        score += 0.1 # Easier to get reasoning right
        
    return min(1.0, max(0.0, score))
