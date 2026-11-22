"""Flower server application entry for Module 1."""

from __future__ import annotations

from sra_fl.server.orchestration import main, run_simulation

__all__ = ["main", "run_simulation"]


if __name__ == "__main__":
    main()
