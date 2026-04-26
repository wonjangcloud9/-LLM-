import re
from pathlib import Path


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)


def build_vocab(text: str) -> dict[str, int]:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return {token: idx for idx, token in enumerate(sorted(set(preprocessed)))}


if __name__ == "__main__":
    text = (Path(__file__).parent / "the-verdict.txt").read_text(encoding="utf-8")
    tokenizer = SimpleTokenizerV1(build_vocab(text))

    sample = '"It\'s the last he painted, you know," Mrs. Gisburn said with pardonable pride.'
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)

    print(f"원문: {sample}")
    print(f"인코딩 ID ({len(ids)}개): {ids}")
    print(f"디코딩: {decoded}")
