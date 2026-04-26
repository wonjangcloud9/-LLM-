from pathlib import Path

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int, stride: int):
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    text = (Path(__file__).parent / "the-verdict.txt").read_text(encoding="utf-8")

    print("=== stride=1, max_length=4, batch=1 (한 토큰씩 미끄러짐) ===")
    loader = create_dataloader_v1(
        text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    it = iter(loader)
    for _ in range(3):
        x, y = next(it)
        print(f"input : {x.tolist()}")
        print(f"target: {y.tolist()}")
        print()

    print("=== stride=4, max_length=4, batch=8 (겹침 없음) ===")
    loader = create_dataloader_v1(
        text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    inputs, targets = next(iter(loader))
    print(f"inputs.shape  = {tuple(inputs.shape)}")
    print(f"targets.shape = {tuple(targets.shape)}")
    print("inputs:")
    print(inputs)
    print("targets:")
    print(targets)
