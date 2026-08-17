import os, json, time, statistics, requests
 
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:8b"
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
 
PROMPTS = [
    "Summarize in 3 bullets for a CFO: 'Revenue grew 12% YoY to $482M. Gross margin held at 71%. Cloud costs rose 9%.'",
    "Write a Python function retry_with_backoff(fn, max_retries=3) using exponential backoff.",
    "Flag one-sided language: 'Client shall indemnify Vendor from all claims, including Vendor's negligence.'",
    "Extract JSON (vendor, invoice_number, amount_due) from: 'Invoice #INV-20481, Acme Cloud Services, $14,250.00.'",
    "As an HR assistant: can an employee expense a client dinner with alcohol?",
]
 
 
def call_ollama(prompt):
    start = time.perf_counter()
    ttft, tokens = None, 0
    try:
        with requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if ttft is None:
                    ttft = time.perf_counter() - start
                tokens += len(chunk.get("response", "").split())
                if chunk.get("done"):
                    break
        duration = time.perf_counter() - start
        return {"model": "Ollama Llama-3-8B", "ttft": ttft, "duration": duration, "tokens": tokens, "tps": tokens / duration}
    except Exception as e:
        return {"model": "Ollama Llama-3-8B", "error": str(e)}
 
 
def call_azure(prompt):
    if not (AZURE_ENDPOINT and AZURE_KEY):
        return {"model": "Azure GPT-4o", "error": "not configured"}
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_VERSION}"
    headers = {"api-key": AZURE_KEY}
    start = time.perf_counter()
    ttft, tokens = None, 0
    try:
        with requests.post(url, headers=headers, json={"messages": [{"role": "user", "content": prompt}], "stream": True}, stream=True, timeout=120) as r:
            for line in r.iter_lines():
                if not line or not line.decode().startswith("data:"):
                    continue
                data = line.decode()[5:].strip()
                if data == "[DONE]":
                    break
                piece = json.loads(data)["choices"][0]["delta"].get("content", "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    tokens += len(piece.split())
        duration = time.perf_counter() - start
        return {"model": "Azure GPT-4o", "ttft": ttft, "duration": duration, "tokens": tokens, "tps": tokens / duration}
    except Exception as e:
        return {"model": "Azure GPT-4o", "error": str(e)}
 
 
def main():
    results = [fn(p) for p in PROMPTS for fn in (call_ollama, call_azure)]
 
    for model in ("Ollama Llama-3-8B", "Azure GPT-4o"):
        ok = [r for r in results if r["model"] == model and "error" not in r]
        print(f"\n{model}: {len(ok)}/5 succeeded")
        if ok:
            print(f"  avg TTFT: {statistics.mean(r['ttft'] for r in ok):.2f}s")
            print(f"  avg duration: {statistics.mean(r['duration'] for r in ok):.2f}s")
            print(f"  avg TPS: {statistics.mean(r['tps'] for r in ok):.1f}")
 
    print("\n| model | ttft | duration | tps |\n|---|---|---|---|")
    for r in results:
        if "error" in r:
            print(f"| {r['model']} | N/A | N/A | N/A |")
        else:
            print(f"| {r['model']} | {r['ttft']:.2f} | {r['duration']:.2f} | {r['tps']:.1f} |")
 
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
 
 
if __name__ == "__main__":
    main()