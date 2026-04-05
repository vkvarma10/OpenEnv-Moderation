import json

def extract_claims(text: str) -> list[str]:
    # In a real environment, this might be a BERT-based claim extraction model
    # Here we mock it based on sentence heuristics for realism
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    return [s for s in sentences if any(w in s.lower() for w in ["is", "are", "causes", "will", "has been", "proven"])]

def verify_source(source_type: str, metadata: dict) -> float:
    # Simulates returning a reliability score 0.0-1.0
    if source_type.lower() == "news":
        base = 0.8
    elif source_type.lower() == "blog":
        base = 0.4
    else:
        base = 0.2

    # Adjust based on follower count or verification if available
    if metadata.get("verified", False):
        base = min(1.0, base + 0.3)
    
    return base

def run_fact_check(claims: list[str]) -> dict:
    # A mock database of known true/false claims
    known_database = {
        "vaccines cause magnetism": "FAKE",
        "the earth is flat": "FAKE",
        "drinking water is good for hydration": "REAL",
        "aliens built the pyramids": "MISLEADING",
    }
    
    results = {}
    for claim in claims:
        # Simple string matching simulation
        verdict = "UNKNOWN"
        for kw, res in known_database.items():
            if kw in claim.lower():
                verdict = res
                break
        results[claim] = verdict
    return results

def calculate_risk_score(claims: list[str], source_reliability: float, text: str) -> float:
    # A heuristic combining signals to get a risk score
    base_risk = sum(1 for c in claims if "cause" in c.lower() or "cure" in c.lower() or "secret" in c.lower()) * 0.2
    
    text_risk = text.lower().count("shocking") * 0.1 + text.lower().count("truth") * 0.1
    
    # High reliability lowers risk
    score = (base_risk + text_risk) * (1.0 - source_reliability)
    
    return min(1.0, max(0.0, score))

def _bert_classifier_mock(text: str) -> str:
    # A mock classification. In reality, a BERT model.
    if "vaccine" in text.lower() and "magnet" in text.lower():
        return "FAKE"
    elif "miracle" in text.lower() and "cure" in text.lower():
        return "HARMFUL"
    elif "expert" in text.lower() and ("maybe" in text.lower() or "reportedly" in text.lower()):
        return "MISLEADING"
    else:
        return "REAL"
