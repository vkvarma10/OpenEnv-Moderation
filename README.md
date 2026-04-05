---
title: OpenEnv
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Moderation Project

A robust, AI-powered misinformation moderation environment compatible with the Meta PyTorch OpenEnv Hackathon standard. This project simulates a structured RL environment for content moderation using a deterministic task grading system, API validations, and Pydantic-based state tracking.

## Features
- **OpenEnv Compliant API**: Exposes `/reset`, `/step`, and `/state` endpoints perfectly formatted for OpenEnv evaluations.
- **Agent Testing Baseline**: Includes a mock agent in `inference.py` to seamlessly test your application before deployment.
- **Pydantic Type Validation**: Strict state, action, and reward models mimicking real-world AI pipelines.
- **Containerized**: Fully set up with Docker for 1-click Hugging Face deployment.

## Installation / Local Execution

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Environment Locally**
   Start the FastAPI container:
   ```bash
   python app.py
   ```
   *Server will run at `http://0.0.0.0:7860`*

3. **Run the Baseline Agent Validation**
   In a separate terminal, test if your environment resolves tasks correctly:
   ```bash
   python inference.py
   ```

## Hugging Face Spaces Deployment Steps 🚀

This repository is strictly designed for **Hugging Face Spaces**. The structure should remain **as is** since the root files (`app.py`, `openenv.yaml`, `Dockerfile`) are scanned directly by evaluation models.

1. Create a new [Hugging Face Space](https://huggingface.co/spaces) and choose **Docker** as the SDK. Choose "Blank" for the Docker template.
2. Upload the entire project to the space repository (Make sure `Dockerfile` is exactly at the root).
3. Go to the space's **Settings > Variables and secrets** and define the following secrets/variables for your inference models if you integrate the agent natively:
   - `HF_TOKEN`: Your Hugging Face Key
   - `API_BASE_URL`: The URL routing your LLM.
   - `MODEL_NAME`: The model used for agent reasoning.
4. Let the Space build. Once it turns to "Running", copy the URL and submit it to the Hackathon portal!

## Folder Structure Notes
- `app.py`: FastAPI server handling HTTP requests and sessions.
- `openenv.yaml`: Critical OpenEnv metadata outlining task difficulties.
- `env/`: The environment logic including `tasks`, `grader`, `environment`, and `models`.
- `inference.py`: Baseline agent script logging standard evaluation commands (`[START]`, `[STEP]`, `[END]`).
