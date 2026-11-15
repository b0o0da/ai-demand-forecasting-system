import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import download

if __name__ == "__main__":
    print(download(force="--force" in sys.argv))
