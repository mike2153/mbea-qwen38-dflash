#!/usr/bin/env python3
"""Speed benchmark for the running server: prefill, decode and DFlash draft acceptance.

stdlib only. Same method as the published numbers:
  decode tok/s  = completion tokens / (total time - time to first token), streamed
  prefill tok/s = prompt tokens / time to first token
  acceptance    = accepted / proposed draft tokens, from /metrics deltas
Every request starts with a fresh nonce so prefix caching cannot fake the prefill figure.

  python3 bench/bench.py [--base http://localhost:8080] [--runs 3] [--long 32000]
"""
import argparse, json, re, statistics, time, urllib.request, uuid
from pathlib import Path

PROMPT = (Path(__file__).parent / 'prompt.txt').read_text()


def metrics(base):
    text = urllib.request.urlopen(base + '/metrics', timeout=20).read().decode()
    out = {'draft': 0.0, 'accepted': 0.0}
    for line in text.splitlines():
        m = re.match(r'^vllm:spec_decode_num_(draft|accepted)_tokens_total(\{[^}]*\})?\s+(\S+)', line)
        if m:
            out[m.group(1)] += float(m.group(3))
    return out


def chat(base, model, user, max_tokens, temperature):
    payload = {'model': model, 'max_tokens': max_tokens, 'temperature': temperature, 'seed': 2718,
               'stream': True, 'stream_options': {'include_usage': True},
               'messages': [{'role': 'system', 'content': f'Benchmark run {uuid.uuid4().hex}.'},
                            {'role': 'user', 'content': user}]}
    req = urllib.request.Request(base + '/v1/chat/completions', json.dumps(payload).encode(),
                                 {'Content-Type': 'application/json'})
    t0, first, usage = time.monotonic(), None, None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            if not raw.startswith(b'data:') or raw[5:].strip() == b'[DONE]':
                continue
            item = json.loads(raw[5:])
            usage = item.get('usage') or usage
            for ch in item.get('choices', []):
                d = ch.get('delta') or {}
                if first is None and (d.get('content') or d.get('reasoning') or d.get('reasoning_content')):
                    first = time.monotonic() - t0
    total = time.monotonic() - t0
    return {'ttft': first, 'prompt': usage['prompt_tokens'], 'completion': usage['completion_tokens'],
            'prefill': usage['prompt_tokens'] / first, 'decode': usage['completion_tokens'] / (total - first)}


def haystack(tokens):
    lines = [f'Record {i:06d}: sensor {i % 97} reported {(i * 7919) % 10007} units at checkpoint {i % 13}.'
             for i in range(tokens // 24)]
    return '\n'.join(lines) + '\n\nWhich sensor reported at checkpoint 5 in record 000005? Answer in one line.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://localhost:8080')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--long', type=int, default=32000, help='long-prompt size in tokens, 0 to skip')
    a = ap.parse_args()
    info = json.load(urllib.request.urlopen(a.base + '/v1/models', timeout=20))['data'][0]
    model = info['id']
    print(f"{model} @ {a.base}  context {info.get('max_model_len')} tokens\n")

    chat(a.base, model, 'Say hi.', 16, 0)  # warm-up
    rows = []
    for temp, label in ((0, 'greedy'), (0.7, 'sampled')):
        m0, runs = metrics(a.base), []
        for i in range(a.runs):
            r = chat(a.base, model, PROMPT, a.max_tokens, temp)
            runs.append(r)
            print(f"  coding task {label} {i + 1}/{a.runs}: {r['completion']} tokens, decode {r['decode']:.1f} tok/s")
        m1 = metrics(a.base)
        acc = (m1['accepted'] - m0['accepted']) / max(m1['draft'] - m0['draft'], 1)
        rows.append((f'Coding task, {label} (x{a.runs})', runs[0]['prompt'],
                     statistics.median(r['prefill'] for r in runs),
                     statistics.median(r['ttft'] for r in runs),
                     statistics.median(r['decode'] for r in runs), f'{acc:.1%}'))
    if a.long:
        r = chat(a.base, model, haystack(a.long), 128, 0)
        print(f"  long prompt: {r['prompt']} tokens, prefill {r['ttft']:.1f} s")
        rows.append((f"Long prompt (~{a.long // 1000}k)", r['prompt'], r['prefill'], r['ttft'], r['decode'], '-'))

    print('\n| Test | Prompt tokens | Prefill tok/s | TTFT s | Decode tok/s | Draft acceptance |')
    print('|---|---:|---:|---:|---:|---:|')
    for name, p, pf, tt, dec, acc in rows:
        print(f'| {name} | {p:,} | {pf:,.0f} | {tt:.2f} | {dec:.1f} | {acc} |')


if __name__ == '__main__':
    main()
