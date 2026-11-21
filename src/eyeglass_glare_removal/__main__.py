"""Module entrypoint enabling `python -m eyeglass_glare_removal`."""

from __future__ import annotations

import sys

from eyeglass_glare_removal.cli import main

if __name__ == "__main__":
    sys.exit(main())
