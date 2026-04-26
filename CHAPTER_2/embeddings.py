from pathlib import Path

import torch

from dataloader import create_dataloader_v1


VOCAB_SIZE = 50257  # GPT-2 BPE 어휘 크기
OUTPUT_DIM = 256    # 임베딩 차원
MAX_LENGTH = 4      # 컨텍스트 길이 (예시용으로 작게)
BATCH_SIZE = 8


def main() -> None:
    text = (Path(__file__).parent / "the-verdict.txt").read_text(encoding="utf-8")

    loader = create_dataloader_v1(
        text,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        stride=MAX_LENGTH,
        shuffle=False,
    )
    inputs, targets = next(iter(loader))
    print(f"inputs.shape  = {tuple(inputs.shape)}   # (배치, 시퀀스)")
    print(f"targets.shape = {tuple(targets.shape)}")

    torch.manual_seed(123)
    token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)

    token_embeddings = token_embedding_layer(inputs)
    print(f"\ntoken_embeddings.shape = {tuple(token_embeddings.shape)}"
          f"   # (배치, 시퀀스, {OUTPUT_DIM}차원)")

    pos_embedding_layer = torch.nn.Embedding(MAX_LENGTH, OUTPUT_DIM)
    pos_embeddings = pos_embedding_layer(torch.arange(MAX_LENGTH))
    print(f"pos_embeddings.shape   = {tuple(pos_embeddings.shape)}"
          f"   # (시퀀스, {OUTPUT_DIM}차원) — 배치 차원 없음")

    input_embeddings = token_embeddings + pos_embeddings
    print(f"input_embeddings.shape = {tuple(input_embeddings.shape)}"
          f"   # 브로드캐스팅으로 합쳐짐")

    print("\n[샘플 0, 위치 0 임베딩 벡터의 앞 8차원]")
    print(f"  토큰      : {token_embeddings[0, 0, :8].tolist()}")
    print(f"  위치      : {pos_embeddings[0, :8].tolist()}")
    print(f"  토큰+위치 : {input_embeddings[0, 0, :8].tolist()}")


if __name__ == "__main__":
    main()
