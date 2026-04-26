from pathlib import Path

path = Path(__file__).parent / "the-verdict.txt"
text = path.read_text(encoding="utf-8")

print(f"총 문자 갯수: {len(text)}")
print(f"처음 99개 문자: {text[:99]}")
