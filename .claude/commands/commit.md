---
description: Conventional Commits 커밋
argument-hint: [추가지시]
---

1. `git status` `git diff --staged` `git diff` `git log -5` 병렬
2. prefix: feat/fix/docs/style/refactor/test/chore/perf/build/ci
3. `<type>: <50자제목>` 명령형,한국어OK
4. 복잡하면 빈 줄 뒤 "왜" 본문
5. 스테이지 비면 관련파일만 add
6. HEREDOC 커밋,푸시X
7. `Co-Authored-By` 트레일러 절대 붙이지 말것

$ARGUMENTS
