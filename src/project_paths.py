"""Local repository paths, resolved before inference scripts change directory.

External repositories default to siblings of this checkout. Set FLASHHEAD_ROOT
or MINIMIND_REPO before starting Python to use another installation. Relative
overrides are resolved against the launch directory; '~' is expanded.
"""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLASHHEAD_ROOT = Path(
    os.environ.get('FLASHHEAD_ROOT') or PROJECT_ROOT.parent / 'SoulX-FlashHead'
).expanduser().resolve()
MINIMIND_REPO = Path(
    os.environ.get('MINIMIND_REPO') or PROJECT_ROOT.parent / 'minimind-o'
).expanduser().resolve()
