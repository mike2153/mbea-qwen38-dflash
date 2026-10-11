#!/usr/bin/env python3
"""Exercise the public API. Writes aggregate checks, never generated text or credentials."""
import argparse
import http.client
import json
from pathlib import Path
import time
import urllib.parse

from request_control import RequestControl, ResponseSocket


def request(base, path, control, body=None, stream=False):
    url = urllib.parse.urlsplit(base)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('validation uses a local HTTP endpoint')
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=10)
    sock = None
    try:
        connection.connect()
        sock = connection.sock
        control.register(sock)
        connection.request('GET' if body is None else 'POST', path,
                           None if body is None else json.dumps(body).encode(),
                           {'Content-Type': 'application/json'})
        connection.sock = ResponseSocket(sock, control, time.monotonic() + 240)
        with connection.getresponse() as response:
            if stream and response.status == 200:
                items, done = [], False
                for line in response:
                    control.check()
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        done = True
                        break
                    items.append(json.loads(data))
                if not done:
                    raise RuntimeError('stream ended without DONE')
                return response.status, items
            data = response.read()
            return response.status, json.loads(data) if data else {}
    except Exception:
        control.requests_complete = False
        raise
    finally:
        connection.close()
        if sock is not None:
            sock.close()
            control.unregister(sock)


def validate(base, control):
    checks = []

    def check(name, passed):
        checks.append({'check': name, 'passed': bool(passed)})
        if not passed:
            raise RuntimeError('API check failed: ' + name)

    status, _ = request(base, '/health', control)
    check('health', status == 200)
    status, models = request(base, '/v1/models', control)
    model = models['data'][0]
    check('context_at_least_180k', status == 200 and model['max_model_len'] >= 180000)
    body = {'model': model['id'], 'messages': [{'role': 'user', 'content':
            'What is 17 times 23? Reply with the number only.'}], 'temperature': 0,
            'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}}
    status, result = request(base, '/v1/chat/completions', control, body)
    check('nonstream_arithmetic', status == 200 and result['choices'][0]['message']['content'].strip() == '391')
    status, result = request(base, '/v1/chat/completions', control,
                             {**body, 'stream': True, 'stream_options': {'include_usage': True}}, True)
    content = ''.join(choice.get('delta', {}).get('content') or ''
                      for item in result for choice in item.get('choices', []))
    usage = [item['usage'] for item in result if item.get('usage')]
    check('stream_content_and_usage', status == 200 and content.strip() == '391'
          and len(usage) == 1 and usage[0]['completion_tokens'] > 0)
    status, result = request(base, '/v1/chat/completions', control,
                             {**body, 'max_tokens': 1024, 'chat_template_kwargs': {'enable_thinking': True}})
    message = result['choices'][0]['message']
    check('thinking_and_answer', status == 200 and bool(message.get('reasoning_content'))
          and '391' in (message.get('content') or ''))
    tool = {'type': 'function', 'function': {'name': 'multiply', 'description': 'Multiply two integers.',
            'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                           'required': ['a', 'b'], 'additionalProperties': False}}}
    prompt = [{'role': 'user', 'content': 'Call multiply with a=17 and b=23. Use the tool.'}]
    status, result = request(base, '/v1/chat/completions', control,
                             {**body, 'messages': prompt, 'tools': [tool], 'tool_choice': 'auto'})
    message = result['choices'][0]['message']
    calls = message.get('tool_calls') or []
    check('tool_call_and_arguments', status == 200 and len(calls) == 1
          and calls[0]['function']['name'] == 'multiply'
          and json.loads(calls[0]['function']['arguments']) == {'a': 17, 'b': 23})
    status, result = request(base, '/v1/chat/completions', control, {**body,
        'messages': prompt + [message, {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '391'}],
        'tools': [tool], 'tool_choice': 'none'})
    check('tool_result_continuation', status == 200 and '391' in (result['choices'][0]['message']['content'] or ''))
    status, _ = request(base, '/v1/chat/completions', control, {**body, 'temperature': -1})
    check('invalid_request_rejected', 400 <= status < 500)
    return {'passed': True, 'checks': checks, 'context': model['max_model_len']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:8080')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cancel-file', type=Path)
    args = parser.parse_args()
    with RequestControl(args.cancel_file) as control:
        result = validate(args.base, control)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f'API validation passed: {len(result["checks"])} checks; {result["context"]:,} context')


if __name__ == '__main__':
    main()
