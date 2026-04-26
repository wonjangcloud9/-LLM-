import sys
import urllib.request
from pathlib import Path


def download(url: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    path, _ = urllib.request.urlretrieve(url, dest)
    return Path(path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download.py <url> [dest]")
        sys.exit(1)

    url = sys.argv[1]
    dest = sys.argv[2] if len(sys.argv) > 2 else Path(url).name
    saved = download(url, dest)
    print(f"Downloaded: {saved.resolve()}")
