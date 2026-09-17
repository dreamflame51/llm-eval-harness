"""Serve the ask page on this machine.

Run:  uv run python scripts/serve.py            http://127.0.0.1:8000
      uv run python scripts/serve.py --port 9000 --no-open

The same thing scripts/ask.py does, with a text box instead of a terminal.
docs/ask.html is the page; this is the twenty lines of server under it.

Why local and not a published artifact: an artifact runs in the viewer's
browser under a content policy that blocks it from reaching anything but a
short list of CDNs - localhost included. A page that talks to the Ollama on
*this* machine has to be served from this machine. That is also the honest
shape of the thing: the model, the index and the corpus are all here, and
nothing about this system is shareable by sending someone a link.

The standard library only, deliberately. A web framework would be the largest
dependency in a repository whose point is measurement, in exchange for
routing two paths.

It binds to 127.0.0.1 and says so: this serves a local model over an
unauthenticated endpoint, and the loopback interface is the whole security
model. Do not move it to 0.0.0.0 without putting something in front.
"""

import argparse
import json
import os
import pathlib
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Same reason as scripts/ask.py: the embedding model is already cached, and
# the Hub check only produces a warning on the way past.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

PAGE = pathlib.Path("docs/ask.html")
MAX_QUESTION = 2000


class Handler(BaseHTTPRequestHandler):
    # One question at a time reaches the model anyway - 4 GB of VRAM does not
    # hold two - so the lock keeps a second browser tab queued rather than
    # thrashing, and makes the wait visible instead of mysterious.
    lock = threading.Lock()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.read_bytes())
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        if self.path != "/ask":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            # AttributeError is in the list because "question" is not
            # guaranteed to be a string: {"question": 5} used to reach .strip()
            # and take the handler down with a traceback instead of a 400.
            question = json.loads(self.rfile.read(length))["question"].strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            self._send(400, "text/plain; charset=utf-8", b"send {'question': '...'}")
            return
        if not question or len(question) > MAX_QUESTION:
            self._send(400, "text/plain; charset=utf-8", b"empty or overlong question")
            return

        from llm_eval_harness import pipeline

        with self.lock:
            try:
                result = pipeline.answer(question)
            except Exception as exc:  # noqa: BLE001 - the browser is the error log here
                message = f"{type(exc).__name__}: {exc}"
                if "onnect" in message:
                    message += "\n\nIs Ollama running? Try: ollama serve"
                self._send(500, "text/plain; charset=utf-8", message.encode())
                return

        payload = {
            "question": question,
            "answer": result["answer"].strip(),
            "contexts": [
                {
                    "source": chunk["source"],
                    "distance": chunk.get("distance"),
                    "text": chunk["text"],
                }
                for chunk in result["contexts"]
            ],
        }
        self._send(200, "application/json; charset=utf-8", json.dumps(payload).encode())

    def _send(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # The default logs every request to stderr, which buries the one line
        # that matters - the address to open.
        if self.path == "/ask":
            print(f"  asked: {args[0] if args else ''}".rstrip(), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser.parse_args()


def main():
    args = parse_args()
    if not PAGE.exists():
        raise SystemExit(f"{PAGE} is missing - run this from the repository root")

    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"{address}   (loopback only - the model is on this machine)")
    print("first question loads the model, about a minute. ctrl-c to stop.\n")
    if not args.no_open:
        threading.Timer(0.5, webbrowser.open, [address]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
