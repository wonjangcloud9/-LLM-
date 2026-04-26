import re
from pathlib import Path

path = Path(__file__).parent / "the-verdict.txt"
text = path.read_text(encoding="utf-8")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_tokens = sorted(set(preprocessed))
vocab = {token: idx for idx, token in enumerate(all_tokens)}

print(f"어휘 사전 크기: {len(vocab)}")
print("처음 30개 토큰 → ID:")
for token, idx in list(vocab.items())[:30]:
    print(f"  {idx:>4}: {token!r}")
