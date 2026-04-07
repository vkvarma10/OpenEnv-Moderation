import os
import yaml
import importlib

def run_validation():
    print("Pre-Submission Checklist Validation Started...")
    
    # 1. HF Space deploys - Automated ping to Space URL
    # Check if app.py has a default @app.get("/")
    print("\n[1] Checking app.py for root health check...")
    with open("server/app.py", "r") as f:
        app_content = f.read()
        if '@app.get("/")' in app_content:
            print("✓ Found root GET endpoint for automated pings.")
        else:
            print("✗ Missing root GET endpoint in app.py!")

    # 2. OpenEnv spec compliance
    print("\n[2] Checking OpenEnv spec compliance...")
    if os.path.exists("openenv.yaml"):
        print("✓ openenv.yaml found at root.")
        try:
            with open("openenv.yaml", "r") as f:
                spec = yaml.safe_load(f)
            # Verify basic keys
            if "environment" in spec and "tasks" in spec:
                print("✓ openenv.yaml properly structured.")
            else:
                print("✗ openenv.yaml is missing required root keys (environment, tasks).")
        except Exception as e:
            print(f"✗ Failed to parse openenv.yaml: {e}")
    else:
        print("✗ openenv.yaml NOT FOUND at root!")

    # Verify typed models
    try:
        models = importlib.import_module("env.models")
        if hasattr(models, "ActionModel") and hasattr(models, "ObservationModel"):
            print("✓ Typed models ActionModel and ObservationModel found.")
        else:
            print("✗ Missing required models in env.models")
    except ImportError:
        print("✗ Cannot import env.models")

    # Verify endpoints in app.py
    if '@app.post("/reset")' in app_content and '@app.post("/step")' in app_content and '@app.get("/state/{session_id}")' in app_content:
        print("✓ Found step()/reset()/state() endpoints in app.py")
    else:
        print("✗ Missing required API endpoints in app.py")

    # 3. Dockerfile builds
    print("\n[3] Checking Dockerfile...")
    if os.path.exists("Dockerfile"):
        print("✓ Dockerfile found at root.")
    else:
        print("✗ Dockerfile NOT FOUND at root!")

    # 4. Baseline reproduces
    print("\n[4] Checking baseline logic and variables...")
    if os.path.exists("llm_client.py"):
        print("✓ llm_client.py found at root.")
        with open("llm_client.py", "r") as f:
            client_content = f.read()
            if "API_BASE_URL" in client_content and "API_KEY" in client_content:
                print("✓ llm_client.py explicitly checks for API_BASE_URL and API_KEY.")
            else:
                print("✗ llm_client.py does not use required LLM variables!")
    else:
        print("✗ llm_client.py NOT FOUND at root!")
        
    if os.path.exists("inference.py"):
        with open("inference.py", "r") as f:
            inf_content = f.read()
            if "llm_client" in inf_content:
                print("✓ inference.py correctly imports from centralized llm_client.py")
            else:
                print("✗ inference.py does NOT import llm_client.py!")
    else:
        print("✗ inference.py NOT FOUND at root!")

    # 5. Tasks with graders
    print("\n[5] Checking tasks and graders...")
    try:
        tasks_m = importlib.import_module("env.tasks")
        tasks = getattr(tasks_m, "TASKS", [])
        if len(tasks) >= 3:
            print(f"✓ Found {len(tasks)} tasks (>= 3).")
            # Verify graders score
            try:
                grader = importlib.import_module("env.grader")
                if hasattr(grader, "grade_trajectory"):
                    print("✓ Grader grade_trajectory found.")
                else:
                    print("✗ Grader logic missing.")
            except ImportError:
                print("✗ Cannot import env.grader")
        else:
            print(f"✗ Found {len(tasks)} tasks, need at least 3.")
    except ImportError:
        print("✗ Cannot import env.tasks")

    print("\nValidation complete!")

if __name__ == "__main__":
    run_validation()
