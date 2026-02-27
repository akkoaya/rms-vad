"""Integration test: simulate a realistic conversation with speech and silence segments."""

import math
import struct

from rms_vad import RmsVAD, VADConfig, VADEventType


SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHUNK_DURATION = CHUNK_SIZE / SAMPLE_RATE  # ~0.064s


def _generate_chunk(amplitude, chunk_size=CHUNK_SIZE, freq=440):
    """Generate a chunk of audio with given amplitude and frequency."""
    samples = []
    for i in range(chunk_size):
        val = int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack("<{}h".format(len(samples)), *samples)


def _generate_silence(chunk_size=CHUNK_SIZE):
    return b"\x00" * (chunk_size * 2)


def test_conversation_flow():
    """Simulate: silence -> speech -> silence -> speech -> silence.

    Verify that SPEECH_START and SPEECH_END events are emitted correctly.
    """
    config = VADConfig(
        threshold=0.05,
        attack=0.1,
        release=0.3,
    )
    vad = RmsVAD(config)

    all_events = []
    t = 0.0

    # Phase 1: 1 second silence (warmup)
    for _ in range(int(1.0 / CHUNK_DURATION)):
        events = vad.feed(_generate_silence(), timestamp=t)
        all_events.extend(events)
        t += CHUNK_DURATION

    # Phase 2: 2 seconds speech (amplitude=15000)
    for _ in range(int(2.0 / CHUNK_DURATION)):
        events = vad.feed(_generate_chunk(15000), timestamp=t)
        all_events.extend(events)
        t += CHUNK_DURATION

    # Phase 3: 1 second silence
    for _ in range(int(1.0 / CHUNK_DURATION)):
        events = vad.feed(_generate_silence(), timestamp=t)
        all_events.extend(events)
        t += CHUNK_DURATION

    # Phase 4: 1 second speech again
    for _ in range(int(1.0 / CHUNK_DURATION)):
        events = vad.feed(_generate_chunk(12000), timestamp=t)
        all_events.extend(events)
        t += CHUNK_DURATION

    # Phase 5: 1 second silence
    for _ in range(int(1.0 / CHUNK_DURATION)):
        events = vad.feed(_generate_silence(), timestamp=t)
        all_events.extend(events)
        t += CHUNK_DURATION

    event_types = [e.type for e in all_events]

    # Should see exactly 2 speech start and 2 speech end events
    start_count = event_types.count(VADEventType.SPEECH_START)
    end_count = event_types.count(VADEventType.SPEECH_END)
    assert start_count == 2, "Expected 2 SPEECH_START, got {}".format(start_count)
    assert end_count == 2, "Expected 2 SPEECH_END, got {}".format(end_count)

    # Verify ordering: START always before END
    starts = [i for i, t in enumerate(event_types) if t == VADEventType.SPEECH_START]
    ends = [i for i, t in enumerate(event_types) if t == VADEventType.SPEECH_END]
    for s, e in zip(starts, ends):
        assert s < e, "SPEECH_START at {} should come before SPEECH_END at {}".format(
            s, e
        )

    # Audio events should exist between each start/end pair
    for s, e in zip(starts, ends):
        segment = event_types[s:e]
        audio_count = segment.count(VADEventType.AUDIO)
        assert audio_count > 0, "No AUDIO events in speech segment"


def test_callback_driven_asr_simulation():
    """Simulate ASR integration using callbacks."""
    asr_log = []

    config = VADConfig(threshold=0.05, attack=0.1, release=0.3)
    vad = RmsVAD(config)

    vad.on_speech_start = lambda pre_buf: asr_log.append(
        ("start", len(pre_buf))
    )
    vad.on_audio = lambda chunk: asr_log.append(("audio", len(chunk)))
    vad.on_speech_end = lambda: asr_log.append(("end",))

    t = 0.0

    # Silence warmup
    for _ in range(20):
        vad.feed(_generate_silence(), timestamp=t)
        t += CHUNK_DURATION

    # Speech
    for _ in range(30):
        vad.feed(_generate_chunk(15000), timestamp=t)
        t += CHUNK_DURATION

    # Silence to trigger end
    for _ in range(20):
        vad.feed(_generate_silence(), timestamp=t)
        t += CHUNK_DURATION

    # Verify callback sequence
    actions = [entry[0] for entry in asr_log]
    assert "start" in actions
    assert "audio" in actions
    assert "end" in actions
    assert actions.index("start") < actions.index("end")
