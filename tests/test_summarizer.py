"""summarizer.py 단위 검사 (마이크·CLI·서버 없이). python3 -m unittest discover tests"""
import importlib.util
import json
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("SUMMARIZER_DEBUG", "0")  # 진행 로그 끔
ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("summarizer", ROOT / "summarizer.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def bare_app() -> "S.Summarizer":
    """__init__(장치·CLI 탐지)을 건너뛴 Summarizer. 검사에 필요한 상태만 채운다."""
    app = object.__new__(S.Summarizer)
    app.lock = threading.Lock()
    app.transcript_ch, app.summary_ch, app.status_ch = S.Channel("transcript"), S.Channel("summary"), S.Channel("status")
    app.config = {"translate": "off", "provider": "off", "language": "ko", "interval": 60}
    app.lines, app.summary, app.summary_tr, app.summary_at = [], "", "", 0.0
    app.summarized_upto, app.started_at = 0, 1000.0
    app.digests, app.digest_upto, app.digest_from = [], 0, 1000.0
    app.saved, app.saved_files = True, {}
    app.recording, app._stt_eof = True, False
    app._stop, app._tr_wake = threading.Event(), threading.Event()
    app._summarizing, app._digesting, app._draft_lock = threading.Lock(), threading.Lock(), threading.Lock()
    app.session = 1
    app.session_dir = Path(tempfile.mkdtemp())
    return app


class FakeProc:
    """apple-stt 대신: 정해 둔 JSON 줄을 내보내고 code 로 끝난다."""

    def __init__(self, events, code=0):
        self.stdout = [json.dumps(e, ensure_ascii=False) + "\n" for e in events]
        self.stdin = Writer()
        self.code = code

    def wait(self):
        return self.code


class Pure(unittest.TestCase):
    def test_split_sentences(self):
        self.assertEqual(S.split_sentences("가입니다. 나는? 다! 라는 아직"), ["가입니다.", "나는?", "다!", "라는 아직"])
        self.assertEqual(S.split_sentences("  "), [])

    def test_parse_bullets(self):
        raw = "- **첫째** 안건\n• 둘째\n\n1) 셋째\n4. 넷째"
        self.assertEqual(S.parse_bullets(raw), ["첫째 안건", "둘째", "셋째"])
        self.assertEqual(S.parse_bullets("3.5% 증가\n2026. 10. 4 배포"), ["3.5% 증가", "2026. 10. 4 배포"])  # 숫자로 시작하는 줄 보존

    def test_char_count_ignores_spaces(self):
        self.assertEqual(S.char_count(" 네 .  "), 2)

    def test_digest_prompt_keeps_braces_in_speech(self):
        text = "중괄호 {name} 를 말함"
        prompt = S.DIGEST_PROMPT.format(prev="(없음)", text=text)
        self.assertIn(text, prompt)
        self.assertIn("<받아쓰기>", prompt)

    def test_render_digests(self):
        t0 = time.mktime((2026, 10, 4, 10, 0, 5, 0, 0, -1))
        lines = [{"id": 0, "t": "10:00:05", "text": "첫  안건은 예산"}, {"id": 1, "t": "10:01:10", "text": "시안은 수요일"}]
        digests = [
            {"id": 0, "start": t0, "end": t0 + 30, "from": 0, "to": 1, "bullets": ["예산 논의"], "tr": ["Budget"]},
            {"id": 1, "start": t0 + 30, "end": t0 + 90, "from": 1, "to": 2, "failed": "시간 초과"},
        ]
        md = S.render_digests(digests, lines, t0)
        self.assertIn("## 10:00\n", md)            # 같은 분이면 시각 하나
        self.assertIn("## 10:00–10:01", md)
        self.assertIn("- 예산 논의\n  - Budget", md)
        self.assertIn("> 요약하지 못했습니다: 시간 초과", md)
        self.assertIn("- 10:00:05 첫 안건은 예산", md)  # 공백 정리


class SummaryTranslation(unittest.TestCase):
    SUMMARY = "## 핵심 논의\n- 예산 20% 증액\n\n## 할 일\n- 팀장: 시안 확정"

    def setUp(self):
        self.orig = S.translate_lines
        S.translate_lines = lambda provider, lang, texts: [f"EN({t})" for t in texts]

    def tearDown(self):
        S.translate_lines = self.orig

    def test_translation_keeps_line_structure(self):
        tr = S.translate_summary("claude", "en", self.SUMMARY)
        self.assertEqual(tr.splitlines(), ["## EN(핵심 논의)", "- EN(예산 20% 증액)", "", "## EN(할 일)", "- EN(팀장: 시안 확정)"])
        self.assertIsNotNone(S.aligned_tr(self.SUMMARY, tr))

    def test_saved_summary_puts_translation_under_each_line(self):
        tr = S.translate_summary("claude", "en", self.SUMMARY)
        self.assertEqual(S.render_summary(self.SUMMARY, tr),
                         "## 핵심 논의 / EN(핵심 논의)\n- 예산 20% 증액\n  - EN(예산 20% 증액)\n\n"
                         "## 할 일 / EN(할 일)\n- 팀장: 시안 확정\n  - EN(팀장: 시안 확정)\n")

    def test_old_whole_translation_goes_below(self):
        self.assertEqual(S.render_summary("## 가\n- 나", "## A"), "## 가\n- 나\n\n---\n\n## A\n")


class Draft(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.saved = (S.APP_DIR, S.DRAFT_PATH)
        S.APP_DIR, S.DRAFT_PATH = self.dir, self.dir / "draft.json"

    def tearDown(self):
        S.APP_DIR, S.DRAFT_PATH = self.saved

    def test_round_trip(self):
        a = bare_app()
        a.lines = [{"id": 0, "t": "10:00:00", "text": "하나", "tr": "one"}, {"id": 1, "t": "10:00:05", "text": "둘"}]
        a.summary, a.summarized_upto, a.saved = "## 핵심 논의\n- 하나", 1, False
        a.digests = [{"id": 0, "start": 1000.0, "end": 1060.0, "from": 0, "to": 1, "bullets": ["하나"]}]
        a.digest_upto, a.digest_from = 1, 1060.0
        a.config["translate"] = "en"
        a._write_draft()
        b = bare_app()
        b._load_draft()
        self.assertEqual(b.lines, a.lines)
        self.assertEqual(b.digests, a.digests)
        self.assertEqual((b.summarized_upto, b.digest_upto, b.digest_from), (1, 1, 1060.0))
        self.assertFalse(b.saved)
        self.assertEqual(b.config["translate"], "en")

    def test_old_draft_without_new_keys(self):
        S.DRAFT_PATH.write_text(json.dumps({"started_at": 5.0, "lines": [{"id": 0, "t": "x", "text": "y"}],
                                            "summary": "s", "summary_tr": "", "translate": "off"}))
        b = bare_app()
        b._load_draft()
        self.assertEqual((b.summarized_upto, b.digest_upto, b.digests), (1, 0, []))
        self.assertFalse(b.saved)  # 저장 여부가 없으면 내용이 있으니 저장 안 된 것으로


class Rounds(unittest.TestCase):
    def setUp(self):
        self.orig = (S.digest, S.summarize)
        self.calls = []

        def fake_digest(provider, prev, text):
            self.calls.append(("digest", text))
            return [text[:10]]

        def fake_summarize(provider, previous, delta):
            self.calls.append(("summary", delta))
            return "## 핵심 논의\n- " + delta[-10:]
        S.digest, S.summarize = fake_digest, fake_summarize

    def tearDown(self):
        S.digest, S.summarize = self.orig

    def app_with(self, *texts):
        app = bare_app()
        app._write_draft = lambda: None
        app.lines = [{"id": i, "t": "10:00:00", "text": t} for i, t in enumerate(texts)]
        return app

    def test_quiet_interval_waits_but_final_takes_the_rest(self):
        app = self.app_with("네.", "내일 배포")  # 공백 빼고 6자
        app._summary_round("claude", final=False)
        app._digest_round("claude", final=False)
        self.assertEqual((self.calls, app.digests, app.summarized_upto), ([], [], 0))
        app._summary_round("claude", final=True)
        app._digest_round("claude", final=True)
        self.assertEqual([c[0] for c in self.calls], ["summary", "digest"])
        self.assertEqual((app.summarized_upto, app.digest_upto, len(app.digests)), (2, 2, 1))

    def test_late_result_from_previous_session_is_dropped(self):
        app = self.app_with("첫 번째 회의에서 나온 아주 긴 이야기입니다")

        def slow_digest(provider, prev, text):
            app.session += 1  # 요약 도중에 새 녹음이 시작됨
            app.lines, app.digests, app.digest_upto = [], [], 0
            return ["옛 회의"]
        S.digest = slow_digest
        app._digest_round("claude", final=True)
        self.assertEqual((app.digests, app.digest_upto), ([], 0))


class Writer:
    """도우미 stdin 대역: 받은 바이트와 닫힘 여부만 기록."""

    def __init__(self):
        self.data, self.closed = b"", False

    def write(self, b):
        if self.closed:
            raise BrokenPipeError
        self.data += b

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeHelper:
    def __init__(self):
        self.stdin = Writer()


class CaptureLoop(unittest.TestCase):
    FRAME = b"\x00\x01" * (S.SAMPLE_RATE * S.FRAME_MS // 1000)

    def test_stop_then_quick_start_does_not_touch_new_recording(self):
        app = bare_app()
        old, new = FakeHelper(), FakeHelper()
        app.stt, app.segments = old, queue.Queue()
        frames = [self.FRAME] * 5

        class Stdout:
            reads = 0

            def read(_self, n):
                Stdout.reads += 1
                if Stdout.reads == 3:  # 멈추자마자 새 녹음 시작: 번호를 올린 뒤 도우미를 바꾼다
                    app.session += 1
                    app.stt, app._stt_eof = new, False
                return frames.pop() if frames else b""

        class FFmpeg:
            stdout, stderr = Stdout(), None
        app.ffmpeg = FFmpeg()
        app._capture_loop()
        self.assertTrue(old.stdin.closed)            # 자기 도우미는 입력 끝
        self.assertFalse(new.stdin.closed)           # 새 녹음의 도우미는 건드리지 않음
        self.assertEqual(new.stdin.data, b"")        # 옛 소리를 새 도우미에 넣지 않음
        self.assertTrue(app.recording)               # 새 녹음을 멈추지 않음
        self.assertFalse(app._stt_eof)

    def test_backlog_replayed_into_restarted_helper(self):
        app = bare_app()
        first, second = FakeHelper(), FakeHelper()
        first.stdin.closed = True                    # 처음 도우미는 이미 죽음
        app.stt, app.segments = first, queue.Queue()
        frames = [bytes([i]) * len(self.FRAME) for i in range(4)]
        feed = list(frames)

        class Stdout:
            reads = 0

            def read(_self, n):
                Stdout.reads += 1
                if Stdout.reads == 3:
                    app.stt = second                 # 다시 뜬 도우미
                return feed.pop(0) if feed else b""

        class FFmpeg:
            stdout, stderr = Stdout(), None
        app.ffmpeg = FFmpeg()
        app.recording = False                        # 사용자가 멈춘 것처럼 끝낸다
        app._capture_loop()
        self.assertEqual(second.stdin.data, b"".join(frames))  # 죽은 동안의 소리까지 순서대로


class AppleReaderSession(unittest.TestCase):
    def test_output_after_new_recording_started_is_ignored(self):
        app = bare_app()
        app._write_draft = lambda: None
        app._summarize_now = lambda final=False: None

        class Proc(FakeProc):
            def __init__(self):
                super().__init__([{"type": "final", "text": "옛 회의 마지막 말."}], code=0)
        proc = Proc()
        app.stt = proc
        real_iter = proc.stdout
        def lines():
            app.session += 1                         # 출력이 오기 전에 새 녹음이 시작됨
            app.lines = []
            yield from real_iter
        proc.stdout = lines()
        app._apple_reader()
        self.assertEqual(app.lines, [])


class AppleReader(unittest.TestCase):
    def run_reader(self, app):
        app._write_draft = lambda: None
        app._summarize_now = lambda final=False: None
        app._apple_reader()
        return [l["text"] for l in app.lines]

    def test_promotes_finished_sentences_before_final(self):
        app = bare_app()
        app._stt_eof = True
        app.stt = FakeProc([
            {"type": "partial", "text": "가입니다. 나"},
            {"type": "partial", "text": "가입니다. 나는"},      # '가입니다.' 가 두 번 같고 뒤 문장이 시작됨 → 줄로
            {"type": "partial", "text": "가입니다. 나는 갑니다. 다"},
            {"type": "final", "text": "가입니다. 나는 갑니다. 다음."},
        ])
        self.assertEqual(self.run_reader(app), ["가입니다.", "나는 갑니다.", "다음."])

    def test_restarts_dead_helper_and_keeps_unfinished_words(self):
        app = bare_app()
        first = FakeProc([{"type": "partial", "text": "끝난 문장. 하던 말"}], code=-9)  # 강제 종료
        second = FakeProc([{"type": "final", "text": "다시 뜬 뒤 문장."}], code=0)
        app.stt = first

        def spawn():
            app._stt_eof = True  # 두 번째 도우미는 입력 끝으로 정상 종료
            return second
        app._spawn_stt = spawn
        events: queue.Queue = queue.Queue()
        app.status_ch.subscribe(events)
        old = S.STT_BACKOFF
        S.STT_BACKOFF = [0]
        try:
            texts = self.run_reader(app)
        finally:
            S.STT_BACKOFF = old
        self.assertEqual(texts, ["끝난 문장.", "하던 말", "다시 뜬 뒤 문장."])
        notices = []
        while not events.empty():
            notices.append(events.get_nowait()[1])
        self.assertTrue(any("다시 시작" in e.get("message", "") for e in notices))

    def test_exit_code_2_stops_without_restart(self):
        app = bare_app()
        app.stt = FakeProc([], code=2)
        app._spawn_stt = lambda: self.fail("exit 2 는 다시 띄우지 않는다")
        stopped = threading.Event()
        app.stop = stopped.set
        self.run_reader(app)
        self.assertTrue(stopped.wait(2))


if __name__ == "__main__":
    unittest.main()
