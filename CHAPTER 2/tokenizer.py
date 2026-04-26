import re
from pathlib import Path

path = Path(__file__).parent / "the-verdict.txt"
text = path.read_text(encoding="utf-8")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

print(f"토큰 갯수: {len(preprocessed)}")
print(preprocessed[:30])
