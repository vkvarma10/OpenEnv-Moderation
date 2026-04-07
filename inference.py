import os
import logging
import json
from openai import OpenAI

logger = logging.getLogger(__name__)

def get_llm_client() -> OpenAI:
    """
    Centralized LLM client for OpenEnv-Moderation.
    STRICTLY complies with the Hackathon's LiteLLM proxy requirements.
    """
    try:
        api_key = os.environ["API_KEY"]
        base_url = os.environ["API_BASE_URL"]
    except KeyError as e:
        error_msg = f"CRITICAL ERROR: Environment variable {e} is missing. API_KEY and API_BASE_URL are strictly required."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info(f"[DEBUG] Validated and using LiteLLM proxy with API_BASE_URL: {base_url}")
    return OpenAI(api_key=api_key, base_url=base_url)

def call_llm_api(prompt: str, model_name: str) -> dict:
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a moderation agent. You must output only a JSON object matching the requested schema. Evaluate step by step, choosing tools first."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        if not content:
            raise json.JSONDecodeError("Empty response content", "", 0)
        
        # Clean response content in case model wraps JSON in markdown blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {content}")
        # Return a fallback decision to prevent system crash
        return {
            "action_type": "MODERATE",
            "classification": "UNKNOWN",
            "confidence": 0.0,
            "action": "ALLOW",
            "justification": f"Parse Error: {str(e)}"
        }
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise

