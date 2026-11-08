"""Entry point: ``python -m bike_network``."""

from .cli import main

# Guarded: spawned worker processes (assignment.assign) re-import this module.
if __name__ == "__main__":
    main()
