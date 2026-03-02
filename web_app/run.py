"""
Launch script for the Sign Language Translator web application.
Run this from the project root directory.
"""

import uvicorn

if __name__ == "__main__":
    print("=" * 50)
    print("  SignLiveAI — Sign Language Translator")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 50)
    uvicorn.run(
        "web_app.backend.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
