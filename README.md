# Realtime Summarizer

회의를 실시간으로 받아적고, 정해 둔 주기마다 요약합니다. 받아적은 문장과 요약은 원하면 다른 언어로 번역합니다.
내 맥에서만 도는 로컬 서버입니다. 요약과 번역은 이미 설치된 AI CLI(Claude Code, Codex 등)를 불러서 처리합니다.
1분 요약 시제품(minute-summary)을 합쳤습니다.

## 실행

```bash
python3 summarizer.py
```

브라우저에서 http://localhost:8792 를 엽니다. 마이크 권한은 서버를 띄운 앱(터미널 등)에 붙습니다.

## 필요한 것

- macOS 26 이상: 온디바이스 실시간 받아쓰기(SpeechAnalyzer)를 씁니다. 처음 실행할 때 `swiftc`로 `bin/apple-stt`를 빌드합니다(Xcode 명령줄 도구 필요).
- ffmpeg: 마이크 녹음에 씁니다.
- whisper-cli(whisper.cpp): 선택입니다. 받아쓰기 모델에서 whisper를 고를 때만 필요합니다.
- 요약·번역용 CLI 중 하나 이상: `claude`, `codex`, `grok`, `cursor-agent-cli`, `agy`, `kiro-cli`, `gemini`, Ollama. 설치된 것을 자동으로 찾습니다. `OPENAI_API_KEY`, `XAI_API_KEY`가 있으면 API도 고를 수 있습니다.
- `claude`는 도구를 끄고 개인 CLAUDE.md·메모리·MCP 없이 부릅니다. 받아적은 말 속의 지시를 실행하지 않게 하기 위해서입니다.

Python은 표준 라이브러리만 씁니다.

## 쓰는 법

- 녹음을 시작하면 말하는 도중의 중간 자막이 1초 간격으로 바뀌고, 문장이 끝나면 한 줄로 올라갑니다.
- 번역을 켜면 줄마다 번역이 붙습니다. CLI에 따라 몇 초에서 20초쯤 걸립니다.
- 요약 칸은 두 가지입니다. "전체 요약"은 주기마다 회의 전체 요약을 다시 씁니다. "구간 요약"은 그 구간에 새로 나온 말만 1~3줄로 정리합니다. 말이 거의 없던 구간은 다음 구간에 합칩니다.
- "지금 요약"을 누르면 둘 다 바로 갱신합니다.
- 받아쓰기 도우미가 도중에 멈추면 다시 띄우고, 그동안의 소리를 이어서 받아적습니다.
- 녹음 중 내용은 `~/Library/Application Support/Realtime Summarizer/draft.json`에 임시로만 둡니다. 서버를 다시 켜도 이 초안을 불러옵니다.
- "저장"을 눌러야 `~/Documents/meetings`에 받아적기·번역·요약·구간 요약 파일이 생깁니다.
- 마이크, 받아쓰기 모델, 요약 CLI, 번역 언어, 요약 주기는 "설정"을 펼쳐 바꿉니다. 펼칠 때 목록을 새로 찾습니다.

## 환경변수

| 이름 | 기본값 | 용도 |
| --- | --- | --- |
| `SUMMARIZER_PORT` | `8792` | 서버 포트 |
| `SUMMARIZER_DATA_DIR` | `~/Documents/meetings` | "저장"으로 남기는 위치 |
| `SUMMARIZER_APP_DIR` | `~/Library/Application Support/Realtime Summarizer` | 임시 초안 위치 |
| `SUMMARIZER_INPUT` | 없음 | 마이크 대신 오디오 파일을 실시간 속도로 넣습니다(테스트용) |
| `SUMMARIZER_CLAUDE_MODEL` | Claude Code 설정 | `claude`로 요약·번역할 때 쓸 모델 (예: `haiku`) |
| `SUMMARIZER_DEBUG` | `1` | `0`이면 진행 로그를 끕니다 |

테스트할 때는 저장 위치와 초안 위치를 따로 잡아 두면 실제 기록과 섞이지 않습니다.

```bash
SUMMARIZER_INPUT=test.wav SUMMARIZER_PORT=8793 SUMMARIZER_DATA_DIR=/tmp/rs-out SUMMARIZER_APP_DIR=/tmp/rs-app python3 summarizer.py
```

## 검사

```bash
python3 -m unittest discover -s tests
```
