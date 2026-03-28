"""Convenience wrapper that keeps the original CLI entrypoint available."""

from app.cli import run


if __name__ == "__main__":
    run()
