# CHAPTER 2 — 텍스트 데이터 다루기

> LLM 은 텍스트를 직접 먹지 못한다. 텍스트를 **숫자 ID 시퀀스**로 바꾸고, 한 토큰씩 어긋난 **(input, target) 쌍**의 텐서 배치로 만들어 줘야 한다. 이 챕터는 그 전처리 파이프라인을 처음부터 만든다.

전체 흐름 한 줄 요약:

```
원문(.txt) → 토큰화 → 어휘 사전 → 토크나이저(encode/decode)
            → BPE(서브워드)  → 슬라이딩 윈도우 → DataLoader 배치
```

---

## 0. 데이터 받기 — [`download.py`](./download.py)

`urllib.request.urlretrieve` 로 임의 URL 의 파일을 받는 작은 스크립트.
이 챕터의 학습 데이터는 Edith Wharton 의 단편 *The Verdict* 전문 (`the-verdict.txt`).

```bash
python download.py <URL> [저장경로]
```

> 한 도메인 텍스트로 시작해서 토크나이저·어휘 사전·DataLoader 가 어떻게 동작하는지 눈으로 확인하기 위함이다. 실제 GPT 학습은 수십 GB 코퍼스를 쓴다.

---

## 1. 데이터 들여다보기 — [`count_chars.py`](./count_chars.py)

먼저 길이부터 본다.

```python
text = path.read_text(encoding="utf-8")
print(len(text))      # 20479 자
print(text[:99])      # "I HAD always thought Jack Gisburn rather a cheap genius--..."
```

전체 **20,479 글자**짜리 텍스트 한 덩어리. 이걸 잘게 쪼갠다.

---

## 2. 정규식 토큰화 — [`tokenizer.py`](./tokenizer.py)

공백 단위로 그냥 자르면 `"hello,"` 와 `"hello"` 가 다른 토큰이 되어 버린다. 그래서 **구두점·`--`·공백을 모두 분리자로** 두고 캡처 그룹으로 토큰 자체도 보존한다.

```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
```

결과:
- 총 **4,690 개 토큰**
- 처음 30개:
  ```
  ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn',
   'rather', 'a', 'cheap', 'genius', '--', 'though', ...]
  ```

`'genius--though'` 가 `'genius'`, `'--'`, `'though'` 세 토큰으로 분해되는 게 핵심.

---

## 3. 어휘 사전 만들기 — [`vocab.py`](./vocab.py)

같은 단어가 매번 다른 ID 면 안 된다. **유일 토큰**만 모아서 정렬한 뒤 0부터 ID 부여.

```python
all_tokens = sorted(set(preprocessed))
vocab = {token: idx for idx, token in enumerate(all_tokens)}
```

결과:
- 어휘 사전 크기: **1,130**
- ID 0~10 = 구두점 (`!`, `"`, `'`, `(`, `)`, `,`, `--`, `.`, `:`, `;`, `?`)
- ID 11 부터 알파벳 순 단어 (`'A'`=11, `'Ah'`=12, ...)

> 이 사전은 *오직 이 소설에 등장한 단어만* 안다. 모르는 단어는 다음 단계에서 처리.

---

## 4. SimpleTokenizerV1 — [`simple_tokenizer.py`](./simple_tokenizer.py)

어휘 사전이 있으면 **encode (텍스트 → ID)**, **decode (ID → 텍스트)** 가 가능해진다.

```python
class SimpleTokenizerV1:
    def encode(self, text):  # 정규식 분리 → vocab 룩업
    def decode(self, ids):   # int_to_str 룩업 → 구두점 앞 공백 제거
```

`"It's the last he painted, ..."` → `[1, 56, 2, 850, 988, ...]` → 거의 원문 복원.

**한계 1**: 사전에 없는 단어가 나오면 `KeyError`.
**한계 2**: 두 문서 사이 경계를 표시할 방법이 없다.

---

## 5. SimpleTokenizerV2 — 특수 토큰 — [`simple_tokenizer_v2.py`](./simple_tokenizer_v2.py)

어휘 사전에 두 개를 추가:
- `<|unk|>` — 모르는 단어 자리 표시자
- `<|endoftext|>` — 문서/구절 경계

```python
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
# encode 시 사전에 없으면 <|unk|> 로 치환
preprocessed = [w if w in str_to_int else "<|unk|>" for w in preprocessed]
```

`"Hello, ... <|endoftext|> ... palace."` 인코딩 결과:
```
Hello → <|unk|>(1131)    palace → <|unk|>(1131)
<|endoftext|> → 1130
```

**여전한 한계**: 모르는 단어가 전부 `<|unk|>` 한 통으로 뭉개진다. 모델이 단어 모양에서 정보를 못 얻는다.

---

## 6. BPE 토크나이저 — [`bpe_tokenizer.py`](./bpe_tokenizer.py)

해법은 **단어 → 자주 등장하는 서브워드 조각** 단위로 쪼개는 것. GPT-2 가 쓰는 BPE 사전을 그대로 가져온다.

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")  # 어휘 50,257
ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
```

전혀 학습한 적 없는 `someunknownPlace` 도 알아서 분해된다:

| ID | 토큰 |
|---|---|
| 617 | `' some'` |
| 34680 | `'unknown'` |
| 27271 | `'Place'` |

`<|unk|>` 가 더 이상 필요 없다. 어떤 단어든 서브워드 조합으로 표현 가능.

---

## 7. 슬라이딩 윈도우 DataLoader — [`dataloader.py`](./dataloader.py)

토큰 ID 시퀀스가 생겼으니, 이제 **"앞부분 → 다음 토큰"** 학습 쌍을 만들어야 한다. 핵심 아이디어:

> **target 은 input 을 정확히 한 토큰 오른쪽으로 시프트한 시퀀스**.

```python
for i in range(0, len(token_ids) - max_length, stride):
    input_chunk  = token_ids[i     : i + max_length]
    target_chunk = token_ids[i + 1 : i + max_length + 1]
```

예 (max_length=4, stride=1):
```
input : [40, 367, 2885, 1464]      = "I HAD always"
target: [367, 2885, 1464, 1807]    = "HAD always thought"
                                     ↑ 한 칸 시프트
```

각 위치에서 모델은 "지금까지 본 토큰들 → 바로 다음 토큰" 을 맞히도록 학습한다.

**파라미터 의미**
- `max_length` — 한 샘플의 토큰 길이 (= context size)
- `stride` — 다음 샘플이 시작하는 간격
  - `stride < max_length` → 윈도우 겹침 (데이터 증강)
  - `stride == max_length` → 겹침·누락 없음 (책 권장 기본값)

`DataLoader` 로 감싸면 `(batch, seq_len)` 모양 텐서가 한 번에 나온다:
```
inputs.shape  = (8, 4)
targets.shape = (8, 4)
```

이 텐서가 바로 다음 단계 **임베딩 레이어** 입력이 된다.

---

## 8. 토큰 임베딩 + 위치 임베딩 — [`embeddings.py`](./embeddings.py)

토큰 ID 정수만 가지고는 모델이 단어 사이의 의미 관계를 표현할 수 없다. 각 ID 를 **학습 가능한 벡터** 로 바꿔준다.

```python
token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, OUTPUT_DIM)  # 50257 × 256
token_embeddings = token_embedding_layer(inputs)
# (8, 4)  →  (8, 4, 256)
```

`nn.Embedding` 은 사실상 **(vocab_size × output_dim) 룩업 테이블** — `inputs` 의 각 정수가 256차원 행 하나로 치환된다.

### 왜 위치 임베딩이 추가로 필요한가

토큰 임베딩만으로는 `"고양이가 쥐를 잡았다"` 와 `"쥐가 고양이를 잡았다"` 가 같은 토큰 집합으로 보인다. 어텐션 자체가 순서를 모르기 때문에, **각 위치(0, 1, 2, ...)** 도 임베딩으로 만들어 더해 준다.

```python
pos_embedding_layer = torch.nn.Embedding(MAX_LENGTH, OUTPUT_DIM)   # 4 × 256
pos_embeddings = pos_embedding_layer(torch.arange(MAX_LENGTH))
# 모양: (4, 256)  ← 배치 차원이 없음

input_embeddings = token_embeddings + pos_embeddings  # 브로드캐스팅
# (8, 4, 256) + (4, 256)  →  (8, 4, 256)
```

| 텐서 | 모양 | 의미 |
|---|---|---|
| `inputs` | `(8, 4)` | 토큰 ID 배치 |
| `token_embeddings` | `(8, 4, 256)` | 각 토큰을 256차원 벡터로 |
| `pos_embeddings` | `(4, 256)` | 위치 0~3 의 벡터 (배치 무관) |
| `input_embeddings` | `(8, 4, 256)` | **모델이 실제로 받는 입력** |

> GPT-2/3 처럼 큰 모델은 `OUTPUT_DIM = 768, 1024, ...` 으로 키우고, `MAX_LENGTH` 도 1024 ~ 수만 토큰으로 늘린다. 여기서는 시각화하기 좋게 256 / 4 로 작게 잡았다.

이 `input_embeddings` 가 다음 챕터 **셀프 어텐션** 의 입력이 된다.

---

## 챕터 2 산출물 정리

| 파일 | 단계 | 출력 형태 |
|---|---|---|
| `download.py` | 데이터 수집 | `the-verdict.txt` |
| `count_chars.py` | 데이터 점검 | 길이 / 미리보기 |
| `tokenizer.py` | 정규식 분리 | `list[str]` (4,690) |
| `vocab.py` | 어휘 사전 | `dict[str, int]` (1,130) |
| `simple_tokenizer.py` | V1 encode/decode | `list[int]` |
| `simple_tokenizer_v2.py` | V2 + 특수 토큰 | `list[int]` (+ `<\|unk\|>`, `<\|endoftext\|>`) |
| `bpe_tokenizer.py` | BPE | `list[int]` (50,257 사전) |
| `dataloader.py` | 슬라이딩 윈도우 | `(inputs, targets)` 텐서 배치 |
| `embeddings.py` | 토큰 + 위치 임베딩 | `(batch, seq, dim)` 입력 임베딩 |

다음 챕터에서는 이 `input_embeddings` 를 받아 **셀프 어텐션** 으로 흘려보낸다.
