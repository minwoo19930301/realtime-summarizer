// apple-stt — 표준입력으로 들어오는 16kHz 모노 s16le PCM을 macOS 온디바이스 받아쓰기(SpeechAnalyzer)로 바꿔
// JSON 줄로 내보낸다. 마이크는 부모(scribe.py의 ffmpeg)가 잡으므로 이 도우미는 마이크 권한이 필요 없다.
//
// 출력(stdout, 한 줄에 JSON 하나):
//   {"type":"ready","locale":"ko_KR"}
//   {"type":"partial","text":"…"}     말하는 도중의 중간 결과 (바뀔 수 있다)
//   {"type":"final","text":"…"}       확정된 문장
//   {"type":"status","message":"…"}   음성 모델 내려받는 중 같은 안내
//   {"type":"error","message":"…"}
// 표준입력이 끝나면(EOF) 남은 소리를 확정해서 내보내고 끝낸다.
//
//   apple-stt [--locale ko-KR]
import AVFoundation
import Foundation
import Speech

let outLock = NSLock()
func emit(_ obj: [String: Any]) {
    guard var data = try? JSONSerialization.data(withJSONObject: obj) else { return }
    data.append(0x0A)
    outLock.lock(); defer { outLock.unlock() }
    data.withUnsafeBytes { raw in
        var off = 0
        while off < raw.count {
            let n = write(STDOUT_FILENO, raw.baseAddress! + off, raw.count - off)
            if n <= 0 { return }
            off += n
        }
    }
}

let args = CommandLine.arguments
let localeId: String = {
    if let i = args.firstIndex(of: "--locale"), i + 1 < args.count { return args[i + 1] }
    return "ko-KR"
}()

let SAMPLE_RATE = 16000.0

@available(macOS 26.0, *)
func run() async -> Int32 {
    let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: localeId)) ?? Locale(identifier: localeId)
    // progressiveTranscription = 실시간 자막용 (중간 결과 + 빠른 응답). 이게 없으면 10초 넘게 모았다가 한꺼번에 낸다.
    let transcriber = SpeechTranscriber(locale: locale, preset: .progressiveTranscription)
    let installed = await SpeechTranscriber.installedLocales
    if !installed.contains(where: { $0.identifier(.bcp47) == locale.identifier(.bcp47) }) {
        do {
            if let req = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                emit(["type": "status", "message": "음성 모델을 내려받는 중입니다 (한 번만)"])
                try await req.downloadAndInstall()
            }
        } catch {
            emit(["type": "error", "message": "음성 모델을 설치하지 못했습니다: \(error.localizedDescription)"])
            return 2
        }
    }
    guard let target = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber]) else {
        emit(["type": "error", "message": "받아쓰기에 쓸 오디오 형식을 찾지 못했습니다"])
        return 2
    }
    guard let source = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: SAMPLE_RATE, channels: 1, interleaved: true),
          let converter = AVAudioConverter(from: source, to: target) else {
        emit(["type": "error", "message": "오디오 형식을 변환할 수 없습니다"])
        return 2
    }

    let analyzer = SpeechAnalyzer(modules: [transcriber])
    try? await analyzer.prepareToAnalyze(in: target)  // 모델을 미리 올려 첫 마디부터 바로 나오게
    let results = Task {
        do {
            for try await r in transcriber.results {
                let text = String(r.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
                if text.isEmpty { continue }
                emit(["type": r.isFinal ? "final" : "partial", "text": text])
            }
        } catch {
            emit(["type": "error", "message": "받아쓰기 중 오류: \(error.localizedDescription)"])
        }
    }

    let (stream, continuation) = AsyncStream.makeStream(of: AnalyzerInput.self)
    do {
        try await analyzer.start(inputSequence: stream)
    } catch {
        emit(["type": "error", "message": "받아쓰기를 시작하지 못했습니다: \(error.localizedDescription)"])
        return 2
    }
    emit(["type": "ready", "locale": locale.identifier])

    // 표준입력 읽기: 0.1초(3200바이트)씩 받아 분석기 형식으로 바꿔 넣는다. EOF면 끝.
    let reader = Thread {
        let chunk = 3200
        var pending = Data()
        let handle = FileHandle.standardInput
        while true {
            let data = handle.readData(ofLength: chunk)
            if data.isEmpty { break }
            pending.append(data)
            let usable = pending.count - pending.count % 2
            if usable < 2 { continue }
            let frames = AVAudioFrameCount(usable / 2)
            guard let inBuf = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: frames) else { continue }
            inBuf.frameLength = frames
            _ = pending.prefix(usable).withUnsafeBytes { raw in
                memcpy(inBuf.int16ChannelData![0], raw.baseAddress!, usable)
            }
            pending.removeFirst(usable)
            let cap = AVAudioFrameCount(Double(frames) * target.sampleRate / SAMPLE_RATE) + 1024
            guard let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: cap) else { continue }
            var fed = false
            var err: NSError?
            let status = converter.convert(to: outBuf, error: &err) { _, outStatus in
                if fed { outStatus.pointee = .noDataNow; return nil }
                fed = true
                outStatus.pointee = .haveData
                return inBuf
            }
            if status != .error, outBuf.frameLength > 0 { continuation.yield(AnalyzerInput(buffer: outBuf)) }
        }
        continuation.finish()
    }
    reader.start()

    // 입력이 끝날 때까지 기다렸다가 남은 소리를 확정한다
    do { try await analyzer.finalizeAndFinishThroughEndOfInput() } catch { await analyzer.cancelAndFinishNow() }
    await results.value
    return 0
}

signal(SIGPIPE, SIG_IGN)
if #available(macOS 26.0, *), SpeechTranscriber.isAvailable {
    exit(await run())
} else {
    emit(["type": "error", "message": "이 Mac에서는 실시간 받아쓰기(macOS 26 SpeechAnalyzer)를 쓸 수 없습니다"])
    exit(2)
}
