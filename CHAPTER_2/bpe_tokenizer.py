import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

print(f"GPT-2 BPE 어휘 크기: {tokenizer.n_vocab}")

text = (
    "Hello, do you like tea? <|endoftext|> "
    "In the sunlit terraces of someunknownPlace."
)

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
decoded = tokenizer.decode(ids)

print(f"\n원문: {text}")
print(f"인코딩 ID ({len(ids)}개): {ids}")
print(f"디코딩: {decoded}")

print("\n[토큰 단위 보기]")
for tid in ids:
    piece = tokenizer.decode([tid])
    print(f"  {tid:>6} → {piece!r}")
