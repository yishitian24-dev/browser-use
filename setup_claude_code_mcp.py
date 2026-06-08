"""
Register browser-use MCP server into Claude Code with one command.

Run from this repo root:
    python setup_claude_code_mcp.py

What it does:
    claude mcp add browser-use -s user -- python -m browser_use.mcp
"""

import subprocess
import sys
from pathlib import Path


def main():
    python = sys.executable
    repo_root = Path(__file__).parent.resolve()

    cmd = [
        "claude", "mcp", "add", "browser-use",
        "-s", "user",
        "--", python, "-m", "browser_use.mcp",
    ]

    print(f"Registering browser-use MCP server with Claude Code...")
    print(f"  Python: {python}")
    print(f"  Repo:   {repo_root}")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Done! Restart Claude Code then run: claude mcp list")
    else:
        print(f"✗ Error: {result.stderr or result.stdout}")
        sys.exit(1)


if __name__ == "__main__":
    main()
