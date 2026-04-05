from env.models import TaskModel

TASKS = [
    TaskModel(
        task_id="task_easy_01",
        difficulty="EASY",
        content="Drinking water is good for hydration, and you should drink at least 8 glasses a day.",
        source_type="news",
        source_reliability=0.9,
        user_reputation=0.9,
        metadata={"verified": True, "date": "2024-01-01"},
        ground_truth_label="REAL",
        ground_truth_action="ALLOW",
        ground_truth_claims=["drinking water is good for hydration"]
    ),
    TaskModel(
        task_id="task_medium_01",
        difficulty="MEDIUM",
        content="Scientists have discovered that aliens built the pyramids! Check out this shocking blog post for the hidden truth they don't want you to see.",
        source_type="blog",
        source_reliability=0.3,
        user_reputation=0.4,
        metadata={"verified": False, "tags": ["conspiracy", "shocking"]},
        ground_truth_label="MISLEADING",
        ground_truth_action="FLAG",
        ground_truth_claims=["aliens built the pyramids"]
    ),
    TaskModel(
        task_id="task_hard_01",
        difficulty="HARD",
        content="A recent expert reportedly stated that vaccines cause magnetism and this will cure all diseases. However, maybe it is just a side effect being studied. We need more research.",
        source_type="user",
        source_reliability=0.1,
        user_reputation=0.2,
        metadata={"verified": False, "shares": 5000, "flagged_by_users": 10},
        ground_truth_label="FAKE", # Contains severe misinformation
        ground_truth_action="REMOVE", # Due to severe harm potential
        ground_truth_claims=["vaccines cause magnetism", "will cure all diseases"]
    )
]

def get_task(task_id: str) -> TaskModel:
    for t in TASKS:
        if t.task_id == task_id:
            return t
    raise ValueError(f"Task {task_id} not found.")
