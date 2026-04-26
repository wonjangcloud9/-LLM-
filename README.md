# 밑바닥부터 만들면서 배우는 LLM

Sebastian Raschka의 *Build a Large Language Model From Scratch* (한국어판: 밑바닥부터 만들면서 배우는 LLM) 챕터별 실습 코드.

## 환경 세팅

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tiktoken torch numpy
```

## 챕터 목차

- [CHAPTER 2 — 텍스트 데이터 다루기](./CHAPTER_2/README.md)
  소설 1편을 받아 토큰화 → 어휘 사전 → BPE → 슬라이딩 윈도우로 학습용 (input, target) 텐서를 만들기까지.
