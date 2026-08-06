from transformers import AutoTokenizer

MODELS = {
    "Qwen":    "Qwen/Qwen2.5-7B-Instruct",
    "Llama 3": "NousResearch/Meta-Llama-3-8B-Instruct",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.1",
}

text = """The Service Provider shall maintain a minimum uptime of 99.95% measured
monthly, excluding scheduled maintenance windows communicated at least seventy-two
(72) hours in advance. In the event that availability falls below the committed
threshold, the Client shall be entitled to a service credit equal to 5% of the
monthly recurring charge for each full percentage point of shortfall, capped at
25% of such charge. All incident tickets classified as Severity-1 must receive an
initial response within fifteen (15) minutes and a resolution plan within four (4)
hours. Notwithstanding the foregoing, remedies under this Section 7.3 shall
constitute the Client's sole and exclusive remedy for availability failures."""

for name, repo in MODELS.items():
    tok = AutoTokenizer.from_pretrained(repo)
    ids = tok(text)["input_ids"]
    print(name, len(ids))