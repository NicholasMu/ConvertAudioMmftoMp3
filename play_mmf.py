"""SMAF (.mmf) file player — parses MMF, decodes ADPCM, synthesizes MIDI, plays via sounddevice.
Also supports converting MMF to MP3/WAV via --convert."""

import os
import struct
import sys

import numpy as np
import sounddevice as sd

# ---------------------------------------------------------------------------
# Constants (mirrors smaf/src/constants.rs)
# ---------------------------------------------------------------------------

CHANNEL_MONO = 0
CHANNEL_STEREO = 1

STREAM_PCM = 0
STREAM_OFFSET_BINARY = 1
STREAM_YAMAHA_ADPCM = 2

PCM_TWOS_COMPLEMENT = 0
PCM_ADPCM = 1

BASE_BIT_4 = 0
BASE_BIT_8 = 1

FORMAT_HANDY_PHONE = 0
FORMAT_MOBILE_COMPRESS = 1
FORMAT_MOBILE_NO_COMPRESS = 2

SAMPLING_FREQ_MAP = {0: 4000, 1: 8000, 2: 11000, 3: 22050, 4: 44100}

TIMEBASE_MAP = {0: 1, 1: 2, 2: 4, 3: 5, 0x10: 10, 0x11: 20, 0x12: 40, 0x13: 50}

# ---------------------------------------------------------------------------
# Low-level parsing helpers (mirrors smaf/src/chunks.rs)
# ---------------------------------------------------------------------------


def parse_timebase(raw: int) -> int:
    return TIMEBASE_MAP.get(raw, 1)


def parse_variable_number(data: memoryview, offset: int) -> tuple[int, int]:
    """Return (value, new_offset)."""
    result = 0
    while True:
        byte = data[offset]
        offset += 1
        result = (result << 7) | (byte & 0x7F)
        if byte & 0x80 == 0:
            break
    return result, offset


def read_u8(data: memoryview, offset: int) -> tuple[int, int]:
    return data[offset], offset + 1


def read_be_u16(data: memoryview, offset: int) -> tuple[int, int]:
    return struct.unpack_from(">H", data, offset)[0], offset + 2


def read_be_u32(data: memoryview, offset: int) -> tuple[int, int]:
    return struct.unpack_from(">I", data, offset)[0], offset + 4


def read_tag(data: memoryview, offset: int) -> tuple[bytes, int]:
    return bytes(data[offset : offset + 4]), offset + 4


def read_chunk(data: memoryview, offset: int) -> tuple[bytes, memoryview, int]:
    """Read a TLV chunk: 4-byte tag + 4-byte length + payload. Returns (tag, payload_view, new_offset)."""
    tag, offset = read_tag(data, offset)
    length, offset = read_be_u32(data, offset)
    payload = data[offset : offset + length]
    return tag, payload, offset + length


# ---------------------------------------------------------------------------
# ADPCM decoder (mirrors smaf_player/src/adpcm.rs)
# ---------------------------------------------------------------------------

STEP_TABLE = [57, 57, 57, 57, 77, 102, 128, 153]


def decode_adpcm(data: bytes | memoryview) -> np.ndarray:
    history = 0
    step_size = 127
    samples = np.empty(len(data) * 2, dtype=np.int16)
    idx = 0

    for byte in data:
        for shift in (4, 0):
            step_val = (byte << shift) & 0xF0
            step_val = (step_val >> 4) & 0x0F
            if step_val & 0x08:
                step_val |= 0x08
            else:
                step_val &= 0x07

            sign = step_val & 8
            delta = step_val & 7
            diff = ((1 + (delta << 1)) * step_size) >> 3
            nstep = (STEP_TABLE[delta] * step_size) >> 6
            nstep = max(127, min(24576, nstep))
            step_size = nstep

            newval = history
            if sign:
                newval -= diff
            else:
                newval += diff
            newval = max(-32768, min(32767, newval))
            history = newval
            samples[idx] = newval
            idx += 1

    return samples[:idx]


# ---------------------------------------------------------------------------
# Score Track sequence parser (mirrors smaf/src/chunks/score_track.rs)
# ---------------------------------------------------------------------------


def parse_score_sequence_mobile(data: memoryview) -> list[tuple[int, dict]]:
    """Parse Mobile Standard sequence data. Returns [(duration, event_dict), ...]."""
    events = []
    offset = 0
    length = len(data)

    while offset < length:
        duration, offset = parse_variable_number(data, offset)
        status, offset = read_u8(data, offset)

        if 0x80 <= status <= 0x8F:
            channel = status & 0x0F
            note, offset = read_u8(data, offset)
            gate_time, offset = parse_variable_number(data, offset)
            events.append((duration, {"type": "note", "channel": channel, "note": note, "velocity": 64, "gate_time": gate_time}))

        elif 0x90 <= status <= 0x9F:
            channel = status & 0x0F
            note, offset = read_u8(data, offset)
            velocity, offset = read_u8(data, offset)
            gate_time, offset = parse_variable_number(data, offset)
            events.append((duration, {"type": "note", "channel": channel, "note": note, "velocity": velocity, "gate_time": gate_time}))

        elif 0xB0 <= status <= 0xBF:
            channel = status & 0x0F
            control, offset = read_u8(data, offset)
            value, offset = read_u8(data, offset)
            events.append((duration, {"type": "control_change", "channel": channel, "control": control, "value": value}))

        elif 0xC0 <= status <= 0xCF:
            channel = status & 0x0F
            program, offset = read_u8(data, offset)
            events.append((duration, {"type": "program_change", "channel": channel, "program": program}))

        elif 0xE0 <= status <= 0xEF:
            channel = status & 0x0F
            lsb, offset = read_u8(data, offset)
            msb, offset = read_u8(data, offset)
            value = ((msb & 0x7F) << 7) | (lsb & 0x7F)
            events.append((duration, {"type": "pitch_bend", "channel": channel, "value": value}))

        elif status == 0xF0:
            exc_len, offset = parse_variable_number(data, offset)
            offset += exc_len

        elif status == 0xFF:
            second, offset = read_u8(data, offset)
            if second == 0x2F:
                offset += 1
                break
            elif second == 0x00:
                pass

        else:
            break

    return events


def parse_score_sequence_handy(data: memoryview) -> list[tuple[int, dict]]:
    """Parse Handy Phone Standard sequence data."""
    events = []
    offset = 0
    length = len(data)

    while offset < length:
        if offset + 4 <= length and data[offset] == 0 and data[offset + 1] == 0 and data[offset + 2] == 0 and data[offset + 3] == 0:
            break

        duration, offset = parse_variable_number(data, offset)
        status, offset = read_u8(data, offset)

        if 0x01 <= status <= 0xFE:
            channel = (status & 0b11000000) >> 6
            octave = (status & 0b00110000) >> 4
            voice = status & 0b00001111
            note_number = 36 + octave * 12 + voice
            gate_time, offset = parse_variable_number(data, offset)
            events.append((duration, {"type": "note", "channel": channel, "note": note_number, "velocity": 64, "gate_time": gate_time}))

        elif status == 0x00:
            next_byte, offset = read_u8(data, offset)
            channel = (next_byte & 0b11000000) >> 6

            if next_byte & 0b00111111 == 0b00110000:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "program_change", "channel": channel, "program": value}))
            elif next_byte & 0b00111111 == 0b00110001:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "bank_select", "channel": channel, "value": value}))
            elif next_byte & 0b00111111 == 0b00110010:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "octave_shift", "channel": channel, "value": value}))
            elif next_byte & 0b00111111 == 0b00110011:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "modulation", "channel": channel, "value": value}))
            elif next_byte & 0b00111111 == 0b00111000:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "pitch_bend", "channel": channel, "value": value}))
            elif next_byte & 0b00110000 == 0b00010000:
                value = next_byte & 0b00001111
                events.append((duration, {"type": "pitch_bend", "channel": channel, "value": value}))
            elif next_byte & 0b00111111 == 0b00110111:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "volume", "channel": channel, "value": value}))
            elif next_byte & 0b00111111 == 0b00111010:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "pan", "channel": channel, "value": value}))
            elif next_byte & 0b00111111 == 0b00111011:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "expression", "channel": channel, "value": value}))
            elif next_byte & 0b00110000 == 0b00000000:
                value = next_byte & 0b00001111
                events.append((duration, {"type": "expression", "channel": channel, "value": value}))
            else:
                continue

        elif status == 0xFF:
            next_byte, offset = read_u8(data, offset)
            if next_byte == 0b11110000:
                exc_len, offset = read_u8(data, offset)
                offset += exc_len
            elif next_byte == 0:
                pass

    return events


# ---------------------------------------------------------------------------
# PCM Audio Track sequence parser (mirrors smaf/src/chunks/pcm_audio_track.rs)
# ---------------------------------------------------------------------------


def parse_pcm_audio_sequence(data: memoryview) -> list[tuple[int, dict]]:
    events = []
    offset = 0
    length = len(data)

    while offset < length:
        if offset + 4 <= length and data[offset] == 0 and data[offset + 1] == 0 and data[offset + 2] == 0 and data[offset + 3] == 0:
            break

        duration, offset = parse_variable_number(data, offset)
        first_byte, offset = read_u8(data, offset)

        if first_byte != 0:
            if first_byte == 0xFF:
                second, offset = read_u8(data, offset)
                if second == 0b11110000:
                    exc_len, offset = read_u8(data, offset)
                    offset += exc_len
                elif second == 0:
                    pass
            else:
                channel = first_byte >> 6
                wave_number = first_byte & 0b00111111
                gate_time, offset = parse_variable_number(data, offset)
                events.append((duration, {"type": "wave_message", "channel": channel, "wave_number": wave_number, "gate_time": gate_time}))
        else:
            second, offset = read_u8(data, offset)
            channel = (second & 0b11000000) >> 6

            if second & 0b00111100 == 0b00110100:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "pitch_bend", "channel": channel, "value": value}))
            elif second & 0b00110000 == 0b00110000:
                value = (second & 0b00001111) * 8
                events.append((duration, {"type": "pitch_bend", "channel": channel, "value": value}))
            elif second & 0b00110111 == 0b00110110:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "volume", "channel": channel, "value": value}))
            elif second & 0b00111010 == 0b00111010:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "pan", "channel": channel, "value": value}))
            elif second & 0b00111011 == 0b00111011:
                value, offset = read_u8(data, offset)
                events.append((duration, {"type": "expression", "channel": channel, "value": value}))
            elif second & 0b00110000 == 0b00000000:
                value = ((second & 0b00001111) - 1) * 31
                events.append((duration, {"type": "expression", "channel": channel, "value": value}))

    return events


# ---------------------------------------------------------------------------
# SMAF file parser
# ---------------------------------------------------------------------------


def parse_smaf(data: memoryview) -> list[tuple[bytes, memoryview]]:
    """Parse top-level MMF file, return list of (tag, payload) chunks."""
    magic = bytes(data[0:4])
    if magic != b"MMMD":
        raise ValueError(f"Not an MMF file (magic: {magic!r})")

    offset = 4
    _file_length, offset = read_be_u32(data, offset)

    chunks = []
    while offset + 8 <= len(data) - 2:
        tag, payload, offset = read_chunk(data, offset)
        chunks.append((tag, payload))

    return chunks


# ---------------------------------------------------------------------------
# Score Track event extraction
# ---------------------------------------------------------------------------


def extract_score_track_note_events(payload: memoryview) -> list[tuple[float, int, int, int, float]]:
    """Extract (time_ms, note, velocity, channel, duration_ms) from ScoreTrack."""
    offset = 0
    format_type, offset = read_u8(payload, offset)
    _sequence_type, offset = read_u8(payload, offset)
    timebase_d_raw, offset = read_u8(payload, offset)
    timebase_g_raw, offset = read_u8(payload, offset)
    timebase_d = parse_timebase(timebase_d_raw)
    timebase_g = parse_timebase(timebase_g_raw)

    if format_type == FORMAT_MOBILE_NO_COMPRESS:
        offset += 16
    elif format_type == FORMAT_HANDY_PHONE:
        offset += 2
    else:
        return []

    sequence_events: list[tuple[int, dict]] = []
    pcm_data: dict[int, tuple[int, int, bytes]] = {}

    while offset < len(payload):
        tag, sub_payload, offset = read_chunk(payload, offset)

        if tag == b"Mtsq":
            if format_type == FORMAT_MOBILE_NO_COMPRESS:
                sequence_events = parse_score_sequence_mobile(sub_payload)
            elif format_type == FORMAT_HANDY_PHONE:
                sequence_events = parse_score_sequence_handy(sub_payload)

        elif tag == b"Mtsp":
            sp_offset = 0
            while sp_offset < len(sub_payload):
                pcmd_tag, pcmd_payload, sp_offset = read_chunk(sub_payload, sp_offset)
                if pcmd_tag.startswith(b"Mwa"):
                    wave_num = pcmd_tag[3]
                    wave_type = pcmd_payload[0]
                    _channel = (wave_type & 0b10000000) >> 7
                    _format = (wave_type & 0b01110000) >> 4
                    _base_bit = wave_type & 0b00001111
                    sampling_freq = struct.unpack_from(">H", pcmd_payload, 1)[0]
                    wave_bytes = bytes(pcmd_payload[3:])
                    pcm_data[wave_num] = (_channel, sampling_freq, wave_bytes)

    result: list[tuple[float, int, int, int, float]] = []
    now = 0.0

    for duration, event in sequence_events:
        t = now
        now += duration * timebase_d

        if event["type"] != "note":
            continue

        note = event["note"]
        if note == 0:
            # PCM wave — handled separately
            continue

        channel = event["channel"]
        velocity = event.get("velocity", 64)
        gate_time = event.get("gate_time", 0)
        note_dur = gate_time * timebase_g

        result.append((t, note, velocity, channel, note_dur))

    return result


def extract_score_track_pcm_events(payload: memoryview) -> list[tuple[float, int, np.ndarray]]:
    """Extract PCM wave events from ScoreTrack (note==0 + Mtsp PCM data)."""
    offset = 0
    format_type, offset = read_u8(payload, offset)
    _sequence_type, offset = read_u8(payload, offset)
    timebase_d_raw, offset = read_u8(payload, offset)
    timebase_g_raw, offset = read_u8(payload, offset)
    timebase_d = parse_timebase(timebase_d_raw)
    timebase_g = parse_timebase(timebase_g_raw)

    if format_type == FORMAT_MOBILE_NO_COMPRESS:
        offset += 16
    elif format_type == FORMAT_HANDY_PHONE:
        offset += 2
    else:
        return []

    sequence_events: list[tuple[int, dict]] = []
    pcm_data: dict[int, tuple[int, int, bytes]] = {}

    while offset < len(payload):
        tag, sub_payload, offset = read_chunk(payload, offset)

        if tag == b"Mtsq":
            if format_type == FORMAT_MOBILE_NO_COMPRESS:
                sequence_events = parse_score_sequence_mobile(sub_payload)
            elif format_type == FORMAT_HANDY_PHONE:
                sequence_events = parse_score_sequence_handy(sub_payload)

        elif tag == b"Mtsp":
            sp_offset = 0
            while sp_offset < len(sub_payload):
                pcmd_tag, pcmd_payload, sp_offset = read_chunk(sub_payload, sp_offset)
                if pcmd_tag.startswith(b"Mwa"):
                    wave_num = pcmd_tag[3]
                    wave_type = pcmd_payload[0]
                    _channel = (wave_type & 0b10000000) >> 7
                    _format = (wave_type & 0b01110000) >> 4
                    _base_bit = wave_type & 0b00001111
                    sampling_freq = struct.unpack_from(">H", pcmd_payload, 1)[0]
                    wave_bytes = bytes(pcmd_payload[3:])
                    pcm_data[wave_num] = (_channel, sampling_freq, wave_bytes)

    result: list[tuple[float, int, np.ndarray]] = []
    now = 0.0

    for duration, event in sequence_events:
        t = now
        now += duration * timebase_d

        if event["type"] != "note":
            continue

        note = event["note"]
        if note != 0:
            continue

        channel = event["channel"]
        wave_num = channel + 1

        if wave_num not in pcm_data:
            continue

        _ch, sr, wave_bytes = pcm_data[wave_num]
        samples = decode_adpcm(wave_bytes)
        result.append((t, sr, samples))

    return result


# ---------------------------------------------------------------------------
# PCM Audio Track event extraction
# ---------------------------------------------------------------------------


def extract_pcm_audio_track_events(payload: memoryview) -> list[tuple[float, int, np.ndarray]]:
    offset = 0
    _format_type, offset = read_u8(payload, offset)
    _sequence_type, offset = read_u8(payload, offset)
    wave_type, offset = read_be_u16(payload, offset)

    _track_channel = (wave_type & 0b1000000000000000) >> 15
    _track_format = (wave_type & 0b0111000000000000) >> 12
    sampling_freq_code = (wave_type & 0b0000111100000000) >> 8
    _base_bit = (wave_type & 0b0000000011110000) >> 4

    track_sr = SAMPLING_FREQ_MAP.get(sampling_freq_code, 8000)

    timebase_d_raw, offset = read_u8(payload, offset)
    timebase_g_raw, offset = read_u8(payload, offset)
    timebase_d = parse_timebase(timebase_d_raw)
    timebase_g = parse_timebase(timebase_g_raw)

    wave_data_map: dict[int, memoryview] = {}
    sequence_events: list[tuple[int, dict]] = []

    while offset < len(payload):
        tag, sub_payload, offset = read_chunk(payload, offset)

        if tag == b"Atsq":
            sequence_events = parse_pcm_audio_sequence(sub_payload)

        elif tag.startswith(b"Awa"):
            wave_num = tag[3]
            wave_data_map[wave_num] = sub_payload

    result: list[tuple[float, int, np.ndarray]] = []
    now = 0.0

    for duration, event in sequence_events:
        t = now
        now += duration * timebase_d

        if event["type"] != "wave_message":
            continue

        wave_num = event["wave_number"]
        if wave_num not in wave_data_map:
            continue

        raw = wave_data_map[wave_num]
        samples = decode_adpcm(raw)
        result.append((t, track_sr, samples))

    return result


# ---------------------------------------------------------------------------
# MIDI Synthesizer — converts note events to PCM
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44100

# Simple FM-synth inspired waveforms for a more pleasant sound
# MIDI program -> waveform type
WAVEFORM_SINE = 0
WAVEFORM_SQUARE = 1
WAVEFORM_SAW = 2
WAVEFORM_TRIANGLE = 3

# Default program waveform mapping
PROGRAM_WAVEFORMS = {
    # Piano-like
    0: WAVEFORM_SINE,
    1: WAVEFORM_SINE,
    2: WAVEFORM_TRIANGLE,
    3: WAVEFORM_SINE,
    # Organ-like
    16: WAVEFORM_SINE,
    17: WAVEFORM_SINE,
    18: WAVEFORM_SINE,
    19: WAVEFORM_SINE,
    # Guitar-like
    24: WAVEFORM_SAW,
    25: WAVEFORM_SAW,
    26: WAVEFORM_SAW,
    # Bass
    32: WAVEFORM_SAW,
    33: WAVEFORM_SAW,
    34: WAVEFORM_SAW,
    # Strings
    40: WAVEFORM_SAW,
    41: WAVEFORM_SAW,
    42: WAVEFORM_SAW,
    # Brass
    56: WAVEFORM_SAW,
    57: WAVEFORM_SAW,
    58: WAVEFORM_SAW,
    # Lead
    80: WAVEFORM_SQUARE,
    81: WAVEFORM_SAW,
    82: WAVEFORM_SINE,
    # Pad
    88: WAVEFORM_SINE,
    89: WAVEFORM_SINE,
}

DEFAULT_WAVEFORM = WAVEFORM_SINE


def midi_note_freq(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def generate_waveform(waveform: int, t: np.ndarray, freq: float) -> np.ndarray:
    phase = 2.0 * np.pi * freq * t
    if waveform == WAVEFORM_SINE:
        return np.sin(phase)
    elif waveform == WAVEFORM_SQUARE:
        return np.sign(np.sin(phase))
    elif waveform == WAVEFORM_SAW:
        return 2.0 * (t * freq - np.floor(0.5 + t * freq))
    elif waveform == WAVEFORM_TRIANGLE:
        return 2.0 * np.abs(2.0 * (t * freq - np.floor(t * freq + 0.5))) - 1.0
    return np.sin(phase)


def apply_envelope(samples: np.ndarray, attack: float, decay: float, sustain: float, release: float, sr: int) -> np.ndarray:
    n = len(samples)
    env = np.ones(n, dtype=np.float32)
    attack_n = int(attack * sr)
    decay_n = int(decay * sr)
    release_n = int(release * sr)

    pos = 0
    # Attack
    if attack_n > 0 and pos < n:
        end = min(attack_n, n)
        env[pos:end] = np.linspace(0.0, 1.0, end - pos)
        pos = end
    # Decay
    if decay_n > 0 and pos < n:
        end = min(pos + decay_n, n)
        env[pos:end] = np.linspace(1.0, sustain, end - pos)
        pos = end
    # Sustain — already 1.0, scale to sustain level for remainder before release
    if sustain < 1.0 and pos < n:
        release_start = max(n - release_n, pos)
        if release_start > pos:
            env[pos:release_start] = sustain
        pos = release_start
    # Release
    if release_n > 0 and pos < n:
        start_level = env[pos - 1] if pos > 0 else sustain
        env[pos:] = np.linspace(start_level, 0.0, n - pos)

    return samples * env


def synthesize_notes(
    note_events: list[tuple[float, int, int, int, float]],
    programs: dict[int, int],
) -> np.ndarray:
    """Render note events into a single PCM float32 buffer."""
    if not note_events:
        return np.array([], dtype=np.float32)

    # Find total duration
    max_end = max(t + dur for t, _note, _vel, _ch, dur in note_events)
    total_samples = int(max_end / 1000.0 * SAMPLE_RATE) + SAMPLE_RATE  # 1s padding

    mix = np.zeros(total_samples, dtype=np.float32)
    t_base = np.arange(total_samples, dtype=np.float32) / SAMPLE_RATE

    for t_ms, note, velocity, channel, dur_ms in note_events:
        if dur_ms <= 0:
            continue

        start_sample = int(t_ms / 1000.0 * SAMPLE_RATE)
        note_samples = int(dur_ms / 1000.0 * SAMPLE_RATE)
        end_sample = min(start_sample + note_samples, total_samples)

        length = end_sample - start_sample
        if length <= 0:
            continue

        freq = midi_note_freq(note)
        amp = (velocity / 127.0) * 0.3  # scale down to avoid clipping with multiple notes

        program = programs.get(channel, 0)
        waveform = PROGRAM_WAVEFORMS.get(program, DEFAULT_WAVEFORM)

        t = t_base[start_sample:end_sample]
        wave = generate_waveform(waveform, t, freq)

        # Add subtle harmonics for richer sound
        wave += 0.3 * generate_waveform(waveform, t, freq * 2.0)
        wave += 0.1 * generate_waveform(waveform, t, freq * 3.0)
        wave /= 1.4

        release_time = min(0.05, dur_ms / 1000.0 * 0.2)
        wave = apply_envelope(wave, attack=0.005, decay=0.02, sustain=0.7, release=release_time, sr=SAMPLE_RATE)

        mix[start_sample:end_sample] += wave * amp

    # Soft-clip
    mix = np.tanh(mix * 1.5) / 1.5

    return mix


# ---------------------------------------------------------------------------
# Mixer — render entire MMF to a single PCM buffer
# ---------------------------------------------------------------------------


def render_mmf(data: memoryview) -> tuple[np.ndarray, int]:
    """Render all tracks to a single float32 PCM buffer at SAMPLE_RATE. Returns (samples, sample_rate)."""
    chunks = parse_smaf(data)

    all_pcm_events: list[tuple[float, int, np.ndarray]] = []
    all_note_events: list[tuple[float, int, int, int, float]] = []
    programs: dict[int, int] = {}

    for tag, payload in chunks:
        if tag.startswith(b"MTR"):
            # PCM wave events (note==0)
            all_pcm_events.extend(extract_score_track_pcm_events(payload))
            # MIDI note events
            all_note_events.extend(extract_score_track_note_events(payload))

        elif tag.startswith(b"ATR"):
            all_pcm_events.extend(extract_pcm_audio_track_events(payload))

    # Find total duration in ms
    end_ms = 0.0

    for t_ms, _sr, samples in all_pcm_events:
        pcm_dur = len(samples) / _sr * 1000.0
        end_ms = max(end_ms, t_ms + pcm_dur)

    for t_ms, _note, _vel, _ch, dur in all_note_events:
        end_ms = max(end_ms, t_ms + dur)

    total_samples = int(end_ms / 1000.0 * SAMPLE_RATE) + SAMPLE_RATE
    mix = np.zeros(total_samples, dtype=np.float32)

    # Mix PCM wave events
    for t_ms, sr_src, samples in all_pcm_events:
        start = int(t_ms / 1000.0 * SAMPLE_RATE)

        # Resample to target SAMPLE_RATE
        if sr_src != SAMPLE_RATE:
            ratio = SAMPLE_RATE / sr_src
            new_len = int(len(samples) * ratio)
            if new_len > 0:
                indices = np.linspace(0, len(samples) - 1, new_len)
                samples = np.interp(indices, np.arange(len(samples)), samples.astype(np.float64)).astype(np.float32)
            else:
                continue

        pcm_f = samples.astype(np.float32) / 32768.0
        end = min(start + len(pcm_f), total_samples)
        length = end - start
        if length > 0:
            mix[start:end] += pcm_f[:length]

    # Synthesize and mix MIDI note events
    if all_note_events:
        synth = synthesize_notes(all_note_events, programs)
        length = min(len(synth), total_samples)
        if length > 0:
            mix[:length] += synth[:length]

    # Soft-clip final mix
    mix = np.tanh(mix * 1.2) / 1.2

    return mix, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


def convert_mmf(filepath: str, output_path: str) -> None:
    import lameenc

    with open(filepath, "rb") as f:
        raw = f.read()

    mv = memoryview(raw)
    mix, sr = render_mmf(mv)

    if len(mix) == 0:
        print("No audio events found in this MMF file.")
        return

    duration = len(mix) / sr
    print(f"Rendered {duration:.1f}s of audio ({len(mix)} samples @ {sr}Hz).")

    # Convert float32 [-1, 1] to int16
    pcm_i16 = np.clip(mix * 32767, -32768, 32767).astype(np.int16)

    if output_path.lower().endswith(".wav"):
        import wave

        with wave.open(output_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_i16.tobytes())
    else:
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(192)
        encoder.set_in_sample_rate(sr)
        encoder.set_channels(1)
        encoder.set_quality(2)  # 2 = high quality

        mp3_data = encoder.encode(pcm_i16.tobytes())
        mp3_data += encoder.flush()

        with open(output_path, "wb") as f:
            f.write(mp3_data)

    print(f"Saved to {output_path} ({os.path.getsize(output_path)} bytes)")


def play_mmf(filepath: str) -> None:
    with open(filepath, "rb") as f:
        raw = f.read()

    mv = memoryview(raw)
    mix, sr = render_mmf(mv)

    if len(mix) == 0:
        print("No audio events found in this MMF file.")
        return

    duration = len(mix) / sr
    print(f"Rendered {duration:.1f}s of audio ({len(mix)} samples @ {sr}Hz). Playing...")
    sd.play(mix, samplerate=sr, blocking=True)
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file.mmf> [output.mp3]")
        print(f"  No output file: play via speakers")
        print(f"  With output file: convert to MP3 or WAV")
        sys.exit(1)

    if len(sys.argv) >= 3:
        convert_mmf(sys.argv[1], sys.argv[2])
    else:
        play_mmf(sys.argv[1])
