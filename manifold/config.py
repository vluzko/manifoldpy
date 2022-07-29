from os import environ
from pathlib import Path

DATA = Path(environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "manifold"
DATA.mkdir(exist_ok=True, parents=True)
CACHE_LOC = DATA / "full_markets.pkl"
