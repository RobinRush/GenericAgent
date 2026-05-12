"""OpenAI-compatible /v1/chat/completions endpoint that exposes the GenericAgent
as a single 'model'. Shared agent instance, streaming supported.

Endpoints:
    POST /v1/chat/completions   (stream and non-stream)
    GET  /v1/models

Run standalone:
    python frontends/oaiapp.py
Or via launcher:
    python launch.pyw --api

Required config in mykey.py:
    oai_api_token = 'sk-your-local-token'
Optional:
    oai_api_host    = '127.0.0.1'   # use '0.0.0.0' to expose on LAN
    oai_api_port    = 18000
    oai_api_timeout = 600           # seconds, per-request
"""
import os, sys, json, time, uuid, threading
from socketserver import ThreadingMixIn
from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from bottle import Bottle, request, response, HTTPResponse, debug as bottle_debug
except ImportError:
    print("[oaiapp] bottle not installed. Run: pip install bottle")
    sys.exit(1)

bottle_debug(True)  # show tracebacks on error pages + log to stderr
import traceback

from agentmain import GeneraticAgent
from llmcore import mykeys

# ---------- single shared agent (same pattern as tgapp.py) ----------
agent = GeneraticAgent()
agent.verbose = False
# NOTE: do NOT touch agent.inc_out — DeltaTracker below handles both modes.

# ---------- config ----------
TOKEN       = mykeys.get('oai_api_token')
HOST        = mykeys.get('oai_api_host', '127.0.0.1')
PORT        = int(mykeys.get('oai_api_port', 18000))
REQ_TIMEOUT = int(mykeys.get('oai_api_timeout', 600))

if not TOKEN:
    print("[oaiapp][WARN] oai_api_token not set in mykey.py — server will refuse all requests.")

app = Bottle()


@app.error(500)
def _err500(err):
    try:
        tb = traceback.format_exc()
        exc = getattr(err, 'exception', None)
        msg = repr(exc) if exc else (getattr(err, 'body', None) or 'unknown')
    except Exception as e:
        tb, msg = f"(handler failure: {e})", "internal error"
    sys.stderr.write(f"[oaiapp][ERROR 500] {msg}\n{tb}\n"); sys.stderr.flush()
    print(f"[oaiapp][ERROR 500] {msg}", flush=True)
    response.content_type = 'application/json'
    return json.dumps({"error": {"message": str(msg), "type": "server_error", "traceback": tb}}, ensure_ascii=False)


# ---------- helpers ----------
def _oai_err(status, msg, etype="invalid_request_error", code=None):
    body = {"error": {"message": msg, "type": etype, "code": code}}
    r = HTTPResponse(status=status, body=json.dumps(body, ensure_ascii=False))
    r.content_type = 'application/json'
    return r


def _check_auth():
    if not TOKEN:
        return _oai_err(503, "oai_api_token not configured on server", "server_error")
    auth = request.headers.get('Authorization', '') or ''
    if not auth.startswith('Bearer ') or auth[7:].strip() != TOKEN:
        return _oai_err(401, "Invalid API key", "authentication_error")
    return None


def _model_name():
    try:
        return agent.get_llm_name()
    except Exception:
        return "genericagent"


def _extract_last_user(messages):
    """Take only the last user message — agent maintains its own history."""
    if not isinstance(messages, list) or not messages:
        return None
    for m in reversed(messages):
        if not isinstance(m, dict) or m.get('role') != 'user':
            continue
        c = m.get('content', '')
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            texts = []
            for part in c:
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'text':
                    texts.append(part.get('text', ''))
                elif part.get('type') == 'image_url':
                    texts.append('[image attached — not yet wired through]')
            return '\n'.join(t for t in texts if t)
        return str(c)
    return None


class DeltaTracker:
    """Robust delta extractor — works whether 'next' is cumulative or incremental."""
    def __init__(self):
        self.emitted = ""

    def feed(self, chunk):
        if not chunk:
            return ""
        if chunk.startswith(self.emitted):     # cumulative form
            delta = chunk[len(self.emitted):]
            self.emitted = chunk
            return delta
        self.emitted += chunk                  # incremental form
        return chunk

    def finalize(self, full_text):
        if not full_text:
            return ""
        if full_text.startswith(self.emitted):
            tail = full_text[len(self.emitted):]
            self.emitted = full_text
            return tail
        return ""


def _sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _chunk(chat_id, model, delta_text=None, finish=None, role=None):
    delta = {}
    if role is not None:
        delta['role'] = role
    if delta_text is not None:
        delta['content'] = delta_text
    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


# ---------- CORS preflight ----------
@app.hook('after_request')
def _cors_after():
    response.set_header('Access-Control-Allow-Origin', '*')
    response.set_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')
    response.set_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')


@app.route('/v1/models', method=['OPTIONS'])
@app.route('/v1/chat/completions', method=['OPTIONS'])
def _options():
    return ''


# ---------- endpoints ----------
@app.route('/v1/models', method='GET')
def list_models():
    err = _check_auth()
    if err:
        return err
    try:
        lst = agent.list_llms()
    except Exception:
        lst = [(0, _model_name(), True)]
    data = [{"id": name, "object": "model", "created": 0, "owned_by": "genericagent"}
            for (_i, name, _cur) in lst]
    response.content_type = 'application/json'
    return json.dumps({"object": "list", "data": data}, ensure_ascii=False)


@app.route('/v1/chat/completions', method='POST')
def chat_completions():
    try:
        err = _check_auth()
        if err:
            return err
        try:
            body = request.json
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
        except Exception as e:
            return _oai_err(400, f"Invalid request body: {e}")

        query = _extract_last_user(body.get('messages', []))
        if not query:
            return _oai_err(400, "No user message found in messages[]")

        stream = bool(body.get('stream', False))
        model = _model_name()
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        print(f"[oaiapp] -> task (stream={stream}, len={len(query)}): {query[:80]!r}", flush=True)

        dq = agent.put_task(query, source='oai_api')

        if stream:
            return _stream_response(dq, chat_id, model)
        return _blocking_response(dq, chat_id, model)
    except Exception as e:
        tb = traceback.format_exc()
        sys.stderr.write(f"[oaiapp][route exc] {e!r}\n{tb}\n"); sys.stderr.flush()
        print(f"[oaiapp][route exc] {e!r}", flush=True)
        return _oai_err(500, f"{type(e).__name__}: {e}", "server_error")


def _blocking_response(dq, chat_id, model):
    tracker = DeltaTracker()
    final = ""
    deadline = time.time() + REQ_TIMEOUT
    try:
        while True:
            remaining = max(1, int(deadline - time.time()))
            item = dq.get(timeout=remaining)
            if 'next' in item:
                tracker.feed(item['next'])
            if 'done' in item:
                final = item.get('done', '') or ''
                tracker.finalize(final)
                break
        final = final or tracker.emitted
    except Exception as e:
        return _oai_err(504, f"Agent timeout/error: {e}", "server_error")

    response.content_type = 'application/json'
    return json.dumps({
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }, ensure_ascii=False)


def _stream_response(dq, chat_id, model):
    response.content_type = 'text/event-stream; charset=utf-8'
    response.set_header('Cache-Control', 'no-cache')
    response.set_header('X-Accel-Buffering', 'no')
    # NOTE: do NOT set 'Connection' — it's hop-by-hop, WSGI servers reject it (wsgiref AssertionError).

    def gen():
        tracker = DeltaTracker()
        try:
            yield _sse(_chunk(chat_id, model, role='assistant'))
            deadline = time.time() + REQ_TIMEOUT
            while True:
                remaining = max(1, int(deadline - time.time()))
                item = dq.get(timeout=remaining)
                if 'next' in item:
                    delta = tracker.feed(item['next'])
                    if delta:
                        yield _sse(_chunk(chat_id, model, delta_text=delta))
                if 'done' in item:
                    tail = tracker.finalize(item.get('done', '') or '')
                    if tail:
                        yield _sse(_chunk(chat_id, model, delta_text=tail))
                    yield _sse(_chunk(chat_id, model, finish='stop'))
                    yield "data: [DONE]\n\n"
                    return
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(f"[oaiapp][stream exc] {e!r}\n{tb}\n"); sys.stderr.flush()
            print(f"[oaiapp][stream exc] {e!r}", flush=True)
            yield _sse(_chunk(chat_id, model, delta_text=f"\n\n[stream error: {type(e).__name__}: {e}]", finish='stop'))
            yield "data: [DONE]\n\n"

    return gen()


# ---------- threaded WSGI server ----------
class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class QuietHandler(WSGIRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[oaiapp] {self.address_string()} - {fmt % args}")


def main():
    # background agent worker (same as agentmain.py __main__)
    threading.Thread(target=agent.run, daemon=True).start()
    try:
        agent.next_llm(0)
    except Exception as e:
        print(f"[oaiapp][WARN] next_llm(0) failed: {e}")

    srv = make_server(HOST, PORT, app, server_class=ThreadingWSGIServer, handler_class=QuietHandler)
    tok_disp = '<configured>' if TOKEN else '<MISSING — set oai_api_token in mykey.py>'
    print(f"[oaiapp] OpenAI-compatible API listening on http://{HOST}:{PORT}")
    print(f"[oaiapp]   POST /v1/chat/completions    GET /v1/models")
    print(f"[oaiapp]   auth: Bearer {tok_disp}")
    print(f"[oaiapp]   model exposed: {_model_name()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[oaiapp] shutting down")


if __name__ == '__main__':
    main()
