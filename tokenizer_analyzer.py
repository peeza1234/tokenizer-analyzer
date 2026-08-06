
from transformers import AutoTokenizer
import time


MODELS = {
    "Llama 3": "NousResearch/Meta-Llama-3-8B-Instruct",
    "Mistral": "mistralai/Mistral-7B-Instruct-v0.1",
    "Qwen":    "Qwen/Qwen2.5-7B-Instruct",
}

MAX_LEN = 128

RUNS = 100

SAMPLE_TEXT = """The Service Provider shall maintain a minimum uptime of 99.95% measured
monthly, excluding scheduled maintenance windows communicated at least seventy-two
(72) hours in advance. In the event that availability falls below the committed
threshold, the Client shall be entitled to a service credit equal to 5% of the
monthly recurring charge for each full percentage point of shortfall, capped at
25% of such charge. All incident tickets classified as Severity-1 must receive an
initial response within fifteen (15) minutes and a resolution plan within four (4)
hours. Notwithstanding the foregoing, remedies under this Section 7.3 shall
constitute the Client's sole and exclusive remedy for availability failures."""


def analyze(name, repo, text):
    tok = AutoTokenizer.from_pretrained(repo)

    tok(text)

    start = time.perf_counter()
    for i in range(RUNS):
        tok(text)
    end = time.perf_counter()

    avg_ms = ((end - start) / RUNS) * 1000

    ids = tok(text)["input_ids"]
    token_count = len(ids)

    return {
        "model": name,
        "tokens": token_count,
        "truncated": token_count > MAX_LEN,
        "latency_ms": avg_ms,
    }


def print_table(rows, max_len):
    header = f"{'Model':<10} {'Tokens':>8} {'Truncated':>11} {'Latency (ms)':>14}"
    print()
    print(f"Sample text tokenized against a {max_len}-token limit")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for r in rows:
        flag = "YES" if r["truncated"] else "no"
        print(
            f"{r['model']:<10} {r['tokens']:>8} {flag:>11} {r['latency_ms']:>14.3f}"
        )

    print("-" * len(header))

    best = min(rows, key=lambda r: r["tokens"])
    worst = max(rows, key=lambda r: r["tokens"])
    overhead = (worst["tokens"] - best["tokens"]) / best["tokens"] * 100
    print(
        f"Most efficient: {best['model']} ({best['tokens']} tokens). "
        f"{worst['model']} needs {overhead:.1f}% more for identical text."
    )
    print()



if __name__ == "__main__":
    results = []

    for name, repo in MODELS.items():
        try:
            results.append(analyze(name, repo, SAMPLE_TEXT))
        except Exception as e:
            print(f"[skipped] {name}: {type(e).__name__}: {e}")

    if results:
        print_table(results, MAX_LEN)