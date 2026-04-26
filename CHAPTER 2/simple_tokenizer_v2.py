import re
from pathlib import Path


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids: list[int]) -> str:
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)


def build_vocab_v2(text: str) -> dict[str, int]:
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {token: idx for idx, token in enumerate(all_tokens)}


if __name__ == "__main__":
    text = (Path(__file__).parent / "the-verdict.txt").read_text(encoding="utf-8")
    vocab = build_vocab_v2(text)
    tokenizer = SimpleTokenizerV2(vocab)

    print(f"어휘 사전 크기: {len(vocab)}")
    print(f"<|endoftext|> ID: {vocab['<|endoftext|>']}")
    print(f"<|unk|>       ID: {vocab['<|unk|>']}")
    print()

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    sample = " <|endoftext|> ".join((text1, text2))

    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)

    print(f"원문: {sample}")
    print(f"인코딩 ID ({len(ids)}개): {ids}")
    print(f"디코딩: {decoded}")
