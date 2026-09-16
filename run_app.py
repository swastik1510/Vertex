import sys
import io

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    port = 8000
    host = "127.0.0.1"
    print("=" * 60)
    print(f"Starting Vertex Fullstack Web App on http://{host}:{port}")
    print("=" * 60)

    # Launch browser automatically if run with --open
    if "--open" in sys.argv:
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
