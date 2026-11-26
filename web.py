"""
web.py  -  test chat window for the Recycar / CarsAndPartsHub parts bot

Runs the same bot as bot.py behind a small web page.

On your own computer:
    python web.py
    then open  http://localhost:8000  in the browser

Settings in ".env" (same file as for bot.py):
    GEMINI_API_KEY=...
    DATABASE_URL=...
    ACCESS_CODE=some-word        optional: testers must type this word to open the chat
    SHOW_SEARCHES=1              optional: 1 = testers can see the database searches (default),
                                           0 = hide them

Needs:  pip install flask
On a host such as Render the start command is:  gunicorn web:app
"""

import hmac
import os
import sys
import threading
import time
import uuid

try:
    from flask import Flask, jsonify, request
except ImportError:
    sys.exit("Flask is missing. Run:  python -m pip install flask")

import bot                      # loads .env and the bot itself

ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()
SHOW_SEARCHES = os.environ.get("SHOW_SEARCHES", "1").strip() not in ("0", "false", "no", "")
MAX_MESSAGE_LENGTH = 600        # characters per customer message
MAX_MESSAGES_PER_CHAT = 40      # protects the free Gemini quota
MAX_OPEN_CHATS = 200
CHAT_LIFETIME = 2 * 60 * 60     # seconds without activity before a chat is forgotten

app = Flask(__name__)

_lock = threading.Lock()
_chats = {}                     # chat id -> {"chat", "lock", "count", "seen"}
_shared = {}                    # Gemini client and instructions, created on first use


def shared():
    with _lock:
        if not _shared:
            if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY is missing")
            instructions, _ = bot.build_instructions()
            _shared["client"] = bot.make_client()
            _shared["instructions"] = instructions
        return _shared


def get_chat(chat_id):
    parts = shared()
    now = time.time()
    with _lock:
        for key in [k for k, v in _chats.items() if now - v["seen"] > CHAT_LIFETIME]:
            del _chats[key]
        entry = _chats.get(chat_id)
        if entry is None:
            if len(_chats) >= MAX_OPEN_CHATS:
                oldest = min(_chats, key=lambda k: _chats[k]["seen"])
                del _chats[oldest]
            entry = {"chat": bot.Chat(parts["client"], parts["instructions"]),
                     "lock": threading.Lock(), "count": 0, "seen": now}
            _chats[chat_id] = entry
        entry["seen"] = now
        return entry


def code_ok(data):
    if not ACCESS_CODE:
        return True
    return hmac.compare_digest(str(data.get("code", "")).strip().encode(), ACCESS_CODE.encode())


@app.get("/")
def page():
    return PAGE, 200, {"Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store"}


@app.get("/health")
def health():
    return "ok"


@app.post("/api/start")
def start():
    data = request.get_json(silent=True) or {}
    if not code_ok(data):
        return jsonify(error="Wrong access code."), 403
    return jsonify(chat_id=uuid.uuid4().hex, greeting=bot.GREETING, show_searches=SHOW_SEARCHES)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    if not code_ok(data):
        return jsonify(error="Wrong access code."), 403
    chat_id = str(data.get("chat_id", ""))[:64]
    message = str(data.get("message", "")).strip()
    if not chat_id or not message:
        return jsonify(error="Empty message."), 400
    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify(error=f"Please keep messages under {MAX_MESSAGE_LENGTH} characters."), 400

    try:
        entry = get_chat(chat_id)
    except Exception as exc:
        return jsonify(error=f"The bot is not set up correctly: {exc}"), 500

    if not entry["lock"].acquire(blocking=False):
        return jsonify(error="Still answering your previous message - one moment."), 429
    try:
        if entry["count"] >= MAX_MESSAGES_PER_CHAT:
            return jsonify(error="This test chat reached its message limit. Start a new chat."), 429
        entry["count"] += 1
        try:
            answer = entry["chat"].reply(message)
        except bot.ModelBusy as exc:
            entry["count"] -= 1                      # a failed message does not use up the limit
            print(f"[model busy] {exc}", flush=True)
            return jsonify(error=bot.BUSY_MESSAGE), 503
        except Exception as exc:
            print(f"[chat error] {str(exc)[:500]}", flush=True)
            return jsonify(error="Sorry, something went wrong on my side. Please try again."), 500
        return jsonify(reply=answer, searches=entry["chat"].trace if SHOW_SEARCHES else [])
    finally:
        entry["lock"].release()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parts assistant - test chat</title>
<style>
  :root { --ink:#1c2430; --muted:#6b7686; --line:#dfe4ea; --bg:#f3f5f8; --card:#ffffff;
          --brand:#b3261e; --brand-ink:#ffffff; --bot:#eef1f5; }
  * { box-sizing:border-box; }
  html, body { height:100%; margin:0; }
  body { font:16px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
         color:var(--ink); background:var(--bg); display:flex; justify-content:center; }
  .app { width:100%; max-width:760px; height:100%; display:flex; flex-direction:column;
         background:var(--card); border-left:1px solid var(--line); border-right:1px solid var(--line); }
  header { padding:14px 18px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:12px; }
  header h1 { font-size:17px; margin:0; font-weight:650; }
  header .tag { font-size:12px; color:var(--brand); border:1px solid var(--brand); border-radius:999px;
                padding:1px 8px; font-weight:600; letter-spacing:.03em; text-transform:uppercase; }
  header .spacer { flex:1; }
  button { font:inherit; cursor:pointer; border-radius:8px; border:1px solid var(--line);
           background:var(--card); color:var(--ink); padding:7px 12px; }
  button:hover { border-color:#b9c1cc; }
  button.primary { background:var(--brand); border-color:var(--brand); color:var(--brand-ink); font-weight:600; }
  button:disabled { opacity:.5; cursor:default; }
  #log { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:86%; padding:10px 13px; border-radius:14px; white-space:pre-wrap; overflow-wrap:anywhere; }
  .msg.bot { background:var(--bot); align-self:flex-start; border-bottom-left-radius:4px; }
  .msg.you { background:var(--ink); color:#fff; align-self:flex-end; border-bottom-right-radius:4px; }
  .msg.note { align-self:center; color:var(--muted); font-size:14px; background:none; padding:2px; text-align:center; }
  .msg.err { align-self:center; color:#8a1c14; background:#fdecea; font-size:14px; }
  .msg a { color:var(--brand); font-weight:600; }
  details.search { align-self:flex-start; max-width:86%; font-size:13px; color:var(--muted);
                   border:1px dashed var(--line); border-radius:10px; padding:6px 10px; }
  details.search summary { cursor:pointer; }
  details.search pre { white-space:pre-wrap; overflow-wrap:anywhere; margin:6px 0 2px;
                       font:12px/1.4 ui-monospace, Consolas, monospace; color:var(--ink); }
  form { display:flex; gap:8px; padding:12px 14px; border-top:1px solid var(--line); }
  input[type=text], input[type=password] { flex:1; font:inherit; padding:10px 12px; border:1px solid var(--line);
           border-radius:10px; min-width:0; }
  input:focus { outline:2px solid var(--brand); outline-offset:1px; border-color:transparent; }
  footer { font-size:12px; color:var(--muted); text-align:center; padding:0 14px 10px; }
  #gate { padding:28px 18px; display:none; flex-direction:column; gap:12px; max-width:380px; margin:auto; }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>CarsAndPartsHub parts assistant</h1><span class="tag">Test</span>
    <span class="spacer"></span>
    <button id="new" type="button">New chat</button>
  </header>

  <div id="gate">
    <strong>Enter the access code to start</strong>
    <input id="code" type="password" autocomplete="off" aria-label="Access code">
    <button id="enter" class="primary" type="button">Open chat</button>
    <span id="gateError" style="color:#8a1c14"></span>
  </div>

  <div id="log" aria-live="polite"></div>
  <form id="form">
    <input id="text" type="text" maxlength="600" autocomplete="off"
           placeholder="e.g. driver side window regulator for a 2016 BMW X5" aria-label="Your message">
    <button id="send" class="primary" type="submit">Send</button>
  </form>
  <footer>Test version. Stock and prices come from a database snapshot and may be out of date.</footer>
</div>

<script>
(function () {
  var log = document.getElementById('log'), form = document.getElementById('form'),
      text = document.getElementById('text'), send = document.getElementById('send'),
      gate = document.getElementById('gate'), codeBox = document.getElementById('code'),
      gateError = document.getElementById('gateError');
  var chatId = null, code = '', showSearches = false, busy = false;

  function add(kind, content) {
    var el = document.createElement('div');
    el.className = 'msg ' + kind;
    if (kind === 'bot') { linkify(el, content); } else { el.textContent = content; }
    log.appendChild(el); log.scrollTop = log.scrollHeight;
    return el;
  }
  // turn web addresses into links without ever inserting raw HTML
  function linkify(el, content) {
    var re = /https?:\/\/[^\s<>"')\]]+/g, last = 0, m;
    while ((m = re.exec(content)) !== null) {
      var url = m[0].replace(/[.,;:!?]+$/, '');
      el.appendChild(document.createTextNode(content.slice(last, m.index)));
      var a = document.createElement('a');
      a.href = url; a.textContent = url; a.target = '_blank'; a.rel = 'noopener noreferrer';
      el.appendChild(a);
      last = m.index + url.length; re.lastIndex = last;
    }
    el.appendChild(document.createTextNode(content.slice(last)));
  }
  function addSearches(list) {
    if (!showSearches || !list || !list.length) return;
    var d = document.createElement('details'); d.className = 'search';
    var s = document.createElement('summary');
    s.textContent = 'Database searches for this answer (' + list.length + ')';
    d.appendChild(s);
    list.forEach(function (item) {
      var pre = document.createElement('pre');
      pre.textContent = item.tool + ' ' + JSON.stringify(item.arguments) + '\n' +
                        (item.found || []).map(function (f) { return '  ' + f; }).join('\n');
      d.appendChild(pre);
    });
    log.appendChild(d); log.scrollTop = log.scrollHeight;
  }
  function post(path, body) {
    return fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                         body: JSON.stringify(body) })
      .then(function (r) { return r.json().then(function (j) { j.httpStatus = r.status; return j; }); });
  }
  function setBusy(state) { busy = state; send.disabled = state; text.disabled = state; if (!state) text.focus(); }

  function start() {
    log.textContent = ''; chatId = null; setBusy(true);
    post('/api/start', { code: code }).then(function (j) {
      if (j.httpStatus === 403) { showGate(code ? 'Wrong access code.' : ''); return; }
      gate.style.display = 'none'; log.style.display = 'flex'; form.style.display = 'flex';
      chatId = j.chat_id; showSearches = !!j.show_searches;
      add('bot', j.greeting); setBusy(false);
    }).catch(function () { add('err', 'Cannot reach the bot. Is web.py still running?'); });
  }
  function showGate(message) {
    gate.style.display = 'flex'; log.style.display = 'none'; form.style.display = 'none';
    gateError.textContent = message || ''; codeBox.focus();
  }
  document.getElementById('enter').addEventListener('click', function () { code = codeBox.value; start(); });
  codeBox.addEventListener('keydown', function (e) { if (e.key === 'Enter') { code = codeBox.value; start(); } });
  document.getElementById('new').addEventListener('click', function () { if (!busy || !chatId) start(); });

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    var message = text.value.trim();
    if (!message || busy || !chatId) return;
    add('you', message); text.value = ''; setBusy(true);
    var wait = add('note', 'Looking that up ...');
    post('/api/chat', { chat_id: chatId, message: message, code: code }).then(function (j) {
      wait.remove();
      if (j.error) { add('err', j.error); } else { addSearches(j.searches); add('bot', j.reply); }
      setBusy(false);
    }).catch(function () { wait.remove(); add('err', 'Connection problem. Please send it again.'); setBusy(false); });
  });

  start();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    on_host = "PORT" in os.environ                 # hosting platforms set PORT
    port = int(os.environ.get("PORT", "8000"))
    if not on_host:
        print(f"Open this address in your browser:  http://localhost:{port}")
        print("Press Ctrl+C here to stop.\n")
    app.run(host="0.0.0.0" if on_host else "127.0.0.1", port=port, threaded=True)
