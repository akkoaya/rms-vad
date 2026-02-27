"""Example: using rms-vad with an ASR module."""

from rms_vad import RmsVAD, VADConfig, VADEventType


def example_event_loop():
    """Event-driven style: iterate over events from an audio stream."""

    vad = RmsVAD(
        VADConfig(
            sample_rate=16000,
            channels=1,
            threshold=0.5,
            attack=0.2,
            release=1.5,
        )
    )

    # Replace with your real audio source (e.g. pyaudio stream)
    def mic_stream():
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
        )
        try:
            while True:
                yield stream.read(1024, exception_on_overflow=False)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    for event in vad.iter_events(mic_stream()):
        if event.type == VADEventType.SPEECH_START:
            print("[VAD] Speech started, pre-buffer frames:", len(event.pre_buffer))
            # asr.start()
            # for frame in event.pre_buffer:
            #     asr.send(frame)

        elif event.type == VADEventType.AUDIO:
            print("[VAD] Audio chunk:", len(event.chunk), "bytes")
            # asr.send(event.chunk)

        elif event.type == VADEventType.SPEECH_END:
            print("[VAD] Speech ended")
            # result = asr.end()
            # print("ASR result:", result)


def example_callback():
    """Callback style: register handlers, then feed audio."""

    vad = RmsVAD(VADConfig(threshold=0.3, attack=0.1, release=1.0))

    @vad_on("speech_start", vad)
    def on_start(pre_buffer):
        print("Speech started! Pre-buffer:", len(pre_buffer), "frames")

    @vad_on("audio", vad)
    def on_audio(chunk):
        pass  # asr.send(chunk)

    @vad_on("speech_end", vad)
    def on_end():
        print("Speech ended!")

    # Feed from any source
    # for chunk in audio_source:
    #     vad.feed(chunk)


def vad_on(event_name, vad):
    """Helper decorator for registering VAD callbacks."""

    def decorator(func):
        if event_name == "speech_start":
            vad.on_speech_start = func
        elif event_name == "audio":
            vad.on_audio = func
        elif event_name == "speech_end":
            vad.on_speech_end = func
        return func

    return decorator


if __name__ == "__main__":
    example_event_loop()
