from os import environ
from pathlib import Path

DATA = Path(environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "manifoldpy"
DATA.mkdir(exist_ok=True, parents=True)
CACHE_LOC = DATA / "full_markets.pkl"
JSON_CACHE_LOC = DATA / "full_markets.json"
