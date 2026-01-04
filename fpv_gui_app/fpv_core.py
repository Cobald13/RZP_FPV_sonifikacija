# fpv_core.py
from __future__ import annotations
import io, re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.signal import butter, filtfilt, medfilt

# --- MIDI export helpers ---
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

def _event_grid_indices(t_sec: np.ndarray, rate_hz: float) -> np.ndarray:
    t_end = float(t_sec[-1])
    if t_end <= 0 or rate_hz <= 0:
        return np.array([0], dtype=int)
    grid = np.arange(0.0, t_end, 1.0/float(rate_hz))
    idx = np.searchsorted(t_sec, grid, side='left')
    idx = np.unique(np.clip(idx, 0, len(t_sec)-1))
    return idx if len(idx) else np.array([0], dtype=int)

def chords_to_midi_dualrate(
    chords_midi, amps, t_sec,
    bpm=120, voices=3, channel=0,
    note_rate_hz=10.0, cc_rate_hz=40.0
) -> MidiFile:
    mid = MidiFile(ticks_per_beat=480)
    tr = MidiTrack(); mid.tracks.append(tr)

    tempo = bpm2tempo(bpm)
    tr.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    tpq = mid.ticks_per_beat
    def sec_to_ticks(dt):
        return int(round(dt * (bpm/60.0) * tpq))

    note_idx = _event_grid_indices(t_sec, note_rate_hz)
    cc_idx   = _event_grid_indices(t_sec, cc_rate_hz)

    events = []
    for i in note_idx: events.append((float(t_sec[i]), 'note', int(i)))
    for i in cc_idx:   events.append((float(t_sec[i]), 'cc',   int(i)))
    events.sort(key=lambda x: (x[0], 0 if x[1]=='cc' else 1))

    last_t = float(t_sec[0])
    last_notes = [None]*voices

    for (cur_t, kind, i) in events:
        dt = cur_t - last_t
        last_t = cur_t
        delta = max(0, sec_to_ticks(dt))
        wrote = False

        if kind == 'cc':
            tr.append(MetaMessage('marker', text='cc', time=delta))
            wrote = True

        else:  # note
            chord = chords_midi[i][:voices]
            vel = int(20 + 100*float(amps[i]))
            vel = max(1, min(127, vel))

            for v, n in enumerate(chord):
                n = int(n)
                if last_notes[v] != n:
                    if last_notes[v] is not None:
                        tr.append(Message('note_off', note=int(last_notes[v]), velocity=0,
                                          time=(delta if not wrote else 0), channel=channel))
                        wrote = True
                        delta = 0
                    tr.append(Message('note_on', note=n, velocity=vel,
                                      time=(delta if not wrote else 0), channel=channel))
                    wrote = True
                    delta = 0
                    last_notes[v] = n

            if not wrote:
                tr.append(MetaMessage('marker', text='t', time=delta))
                wrote = True

        if not wrote:
            tr.append(MetaMessage('marker', text='t', time=delta))

    for v, n in enumerate(last_notes):
        if n is not None:
            tr.append(Message('note_off', note=int(n), velocity=0, time=0, channel=channel))
    return mid

# --- throttle segmentation (reuse from notebook) ---
def _lowpass_1d(x, cutoff, fs, order=3):
    b, a = butter(order, min(0.999, cutoff/(0.5*fs)), btype='low')
    return filtfilt(b, a, x)

def _pick_throttle_vec(df: pd.DataFrame) -> np.ndarray:
    for c in ["rcCommands[3]","rcCommand[3]","setpoint[3]"]:
        if c in df.columns:
            v = df[c].to_numpy().astype(float)
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            return np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
    mcols = [c for c in df.columns if c.startswith("motor[")]
    if mcols:
        v = df[mcols].to_numpy().mean(axis=1)
        return (v - v.min()) / (v.ptp() + 1e-12)
    raise ValueError("Ne najdem throttle stolpca (rcCommands[3]/motor[...]).")

def get_throttle_vector(df: pd.DataFrame, fs: float, lp_cut=3.0) -> np.ndarray:
    thr = _pick_throttle_vec(df)
    return np.clip(_lowpass_1d(thr, lp_cut, fs), 0, 1)

def segment_by_hysteresis(x, fs, lo=0.35, hi=0.45, min_len_s=0.30, nbins=4):
    edges = np.linspace(0, 1, nbins+1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    q = np.digitize(x, edges[1:-1])
    target = centers[q]

    segs = []
    start = 0
    cur = target[0]
    for i in range(1, len(target)):
        if (cur <= lo and target[i] >= hi) or (cur >= hi and target[i] <= lo) or (abs(target[i]-cur) > (hi-lo)):
            segs.append((start, i, float(cur)))
            start = i
            cur = target[i]
    segs.append((start, len(target), float(cur)))

    min_len = int(min_len_s * fs)
    merged = []
    for s, e, v in segs:
        if not merged:
            merged.append([s, e, v]); continue
        if (e - s) < min_len:
            merged[-1][1] = e
        else:
            merged.append([s, e, v])
    return [(s, e, v) for s, e, v in merged if e > s]

def segments_to_midi(segs, chords_midi, amps, t_sec, bpm=60, voices=3, channel=0) -> MidiFile:
    mid = MidiFile()
    tr = MidiTrack(); mid.tracks.append(tr)

    tpq = mid.ticks_per_beat
    tempo = bpm2tempo(bpm)
    tr.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    def ticks_for_dt(dt_sec):
        beats = (dt_sec * 1e6) / tempo
        return int(beats * tpq)

    last = [None]*voices

    for (s, e, _lvl) in segs:
        dur_s = float(t_sec[e-1] - t_sec[s])
        chord = chords_midi[s][:voices]
        vel = int(20 + 100 * float(amps[s]))
        vel = max(1, min(127, vel))

        for v, n in enumerate(chord):
            n = int(n)
            if last[v] != n:
                if last[v] is not None:
                    tr.append(Message('note_off', note=int(last[v]), velocity=0, time=0, channel=channel))
                tr.append(Message('note_on', note=n, velocity=vel, time=0, channel=channel))
                last[v] = n

        tr.append(MetaMessage('marker', text='seg', time=ticks_for_dt(dur_s)))

    for v, n in enumerate(last):
        if n is not None:
            tr.append(Message('note_off', note=int(n), velocity=0, time=0, channel=channel))
    return mid

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
SCALES = {
    'major':[0,2,4,5,7,9,11],
    'minor':[0,2,3,5,7,8,10],
    'dorian':[0,2,3,5,7,9,10],
    'mixolydian':[0,2,4,5,7,9,10],
    'lydian':[0,2,4,6,7,9,11],
    'phrygian':[0,1,3,5,7,8,10],
    'locrian':[0,1,3,5,6,8,10],
    'harm_minor':[0,2,3,5,7,8,11],
    'mel_minor':[0,2,3,5,7,9,11],
}

def load_blackbox_csv_bytes(uploaded_bytes: bytes) -> tuple[pd.DataFrame, float]:
    if not isinstance(uploaded_bytes, (bytes, bytearray)):
        uploaded_bytes = bytes(uploaded_bytes)

    text = uploaded_bytes.decode('utf-8', errors='ignore')
    lines = text.splitlines(keepends=True)

    hdr_idx = next(i for i, l in enumerate(lines)
                   if 'loopIteration' in l or 'loopiteration' in l.lower())
    df = pd.read_csv(io.StringIO(''.join(lines[hdr_idx:])), engine='python')

    rx = re.compile(r'^\s*"?looptime"?\s*,\s*([0-9]+)\s*$', re.IGNORECASE)
    looptime_us = None
    for i in range(min(300, len(lines))):
        m = rx.match(lines[i])
        if m:
            looptime_us = int(m.group(1))
            break
    if looptime_us is None:
        raise ValueError("Ni 'looptime' v CSV glavi.")

    fs = 1.0 / (looptime_us * 1e-6)
    return df, fs

def load_blackbox_csv(path: str) -> tuple[pd.DataFrame, float]:
    with open(path, "rb") as f:
        return load_blackbox_csv_bytes(f.read())

def get_t_sec(df: pd.DataFrame) -> np.ndarray:
    t_us = df["time"].to_numpy(dtype=np.float64)
    return (t_us - t_us[0]) * 1e-6

def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manjkajo stolpci: {missing}")

def clean1d(x):
    x = np.asarray(x, float)
    if np.isnan(x).any():
        n = np.isnan(x)
        if n.all():
            return np.zeros_like(x)
        x[n] = np.interp(np.flatnonzero(n), np.flatnonzero(~n), x[~n])
    return x

def lowpass(x, cutoff, fs, order=3):
    wn = min(0.999, float(cutoff) / (0.5 * float(fs)))
    b, a = butter(order, wn, btype='low')
    return filtfilt(b, a, x)

def norm01(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

def midi_to_hz(m):
    m = np.asarray(m, float)
    return 440.0 * (2 ** ((m - 69.0) / 12.0))

def build_scale_set(root='A', scale='minor'):
    root_idx = NOTE_NAMES.index(root)
    pcs = [(root_idx + deg) % 12 for deg in SCALES[scale]]
    return set(pcs), pcs

def quantize_midi_to_key(m, pcs_set):
    cands = [int(np.floor(m)) + d for d in range(-12, 13)]
    # najprej kaznuj note izven lestvice, potem šele razdalja
    best = min(cands, key=lambda x: (0 if (x % 12) in pcs_set else 1, abs(x - m)))
    return best

def chord_from_root(midi_root, chord_type='triad', voices=3):
    pc = midi_root % 12
    is_major = pc in {0,2,5,7,9}
    third = 4 if is_major else 3

    r = int(midi_root)
    t = int(midi_root + third) + 12
    f = int(midi_root + 7) + 12
    chord = [r, t, f]

    if chord_type in ('7', 'maj7'):
        sev = int(midi_root + (11 if chord_type == 'maj7' else 10)) + 12
        chord.append(sev)

    if chord_type == 'sus2':
        chord = [r, r + 2 + 12, r + 7 + 12]
    elif chord_type == 'sus4':
        chord = [r, r + 5 + 12, r + 7 + 12]

    return chord[:int(voices)]

def smooth_pitch_continuous(pitch_midi, fs, med_k=9, lp_hz=2.0):
    x = clean1d(np.asarray(pitch_midi, float))
    if med_k is not None and int(med_k) >= 3:
        k = int(med_k)
        if k % 2 == 0:
            k += 1
        x = medfilt(x, kernel_size=k)
    if lp_hz is not None and float(lp_hz) > 0:
        x = lowpass(x, cutoff=float(lp_hz), fs=fs, order=3)
    return x

def debounce_notes(quant_midi, t_sec, hold_ms=120, min_step=2):
    q = np.asarray(quant_midi, int)
    t = np.asarray(t_sec, float)
    if len(q) == 0:
        return q

    hold_s = max(0.0, float(hold_ms) / 1000.0)
    min_step = int(min_step)

    out = q.copy()
    cur = int(q[0])
    last_change_t = float(t[0])

    for i in range(1, len(q)):
        cand = int(q[i])
        dt = float(t[i]) - last_change_t

        big_enough = (abs(cand - cur) >= min_step) if min_step > 0 else (cand != cur)
        long_enough = (dt >= hold_s) if hold_s > 0 else True

        if cand != cur and big_enough and long_enough:
            cur = cand
            last_change_t = float(t[i])
        out[i] = cur

    return out

@dataclass
class ComputeConfig:
    root: str = 'A'
    scale: str = 'minor'
    chord_type: str = '7'
    voices: int = 3
    lp_cut: float = 30.0
    pitch_med_k: int = 9
    pitch_lp_hz: float = 2.0
    hold_ms: int = 120
    min_step: int = 2
    midi_min: int = 52
    midi_max: int = 76
    curve: float = 0.65
    octave_shift: int = 1

def prepare_params_and_chords(df: pd.DataFrame, fs: float, cfg: ComputeConfig) -> dict:
    ensure_cols(df, ["gyroADC[0]","gyroADC[1]","gyroADC[2]",
                     "accSmooth[0]","accSmooth[1]","accSmooth[2]"])

    gyro = df[["gyroADC[0]","gyroADC[1]","gyroADC[2]"]].to_numpy()
    acc  = df[["accSmooth[0]","accSmooth[1]","accSmooth[2]"]].to_numpy()

    pid_cols = [c for c in ["axisError[0]","axisError[1]","axisError[2]"] if c in df.columns]
    pid_error = df[pid_cols].to_numpy() if len(pid_cols)==3 else np.zeros((len(df),3))

    erpm_cols = [c for c in df.columns if c.startswith("eRPM[")]
    rpm_avg = (df[erpm_cols].to_numpy()/7.0).mean(axis=1) if erpm_cols else np.zeros(len(df))

    acc_lp = np.column_stack([clean1d(lowpass(acc[:,i], cfg.lp_cut, fs)) for i in range(3)])
    gyro_mag = clean1d(np.linalg.norm(gyro, axis=1))
    pid_mag  = clean1d(np.linalg.norm(pid_error, axis=1))

    amp    = norm01(acc_lp[:,2])
    timbre = norm01(clean1d(rpm_avg))
    vibr   = norm01(pid_mag) * 0.2
    pan    = norm01(acc_lp[:,0])

    g = norm01(gyro_mag)
    pitch_midi_raw = cfg.midi_min + (g**cfg.curve) * (cfg.midi_max - cfg.midi_min)

    pitch_midi_smooth = smooth_pitch_continuous(
        pitch_midi_raw, fs,
        med_k=cfg.pitch_med_k,
        lp_hz=cfg.pitch_lp_hz
    )

    pcs_set, _ = build_scale_set(cfg.root, cfg.scale)
    quant_midi = np.array([quantize_midi_to_key(m, pcs_set) for m in pitch_midi_smooth], dtype=int)
    quant_midi = quant_midi + 12 * int(cfg.octave_shift)

    t_sec = get_t_sec(df)
    quant_midi = debounce_notes(quant_midi, t_sec, hold_ms=cfg.hold_ms, min_step=cfg.min_step)

    chords_m = [chord_from_root(int(m), cfg.chord_type, cfg.voices) for m in quant_midi]
    chords_h = [list(map(float, midi_to_hz(ch))) for ch in chords_m]

    return dict(
        t_sec=t_sec,
        pitch_midi_raw=pitch_midi_raw,
        pitch_midi=pitch_midi_smooth,
        quant_midi=quant_midi,
        amp=amp, timbre=timbre, vibr=vibr, pan=pan,
        chords_midi=chords_m, chords_hz=chords_h
    )

def resample_to_fps(arr: np.ndarray, t_sec: np.ndarray, fps: int) -> tuple[np.ndarray, np.ndarray]:
    t_src = np.asarray(t_sec, float)
    n_frames = int(round(t_src[-1] * fps))
    t_dst = np.linspace(0, t_src[-1], max(1, n_frames))
    return np.interp(t_dst, t_src, np.asarray(arr, float)), t_dst

def chords_to_fps(chords_hz: list[list[float]], t_sec: np.ndarray, fps: int, voices: int = 3) -> tuple[np.ndarray, np.ndarray]:
    t_src = np.asarray(t_sec, float)
    n_frames = int(round(t_src[-1] * fps))
    t_dst = np.linspace(0, t_src[-1], max(1, n_frames))

    idx = np.searchsorted(t_src, t_dst, side='left')
    idx = np.clip(idx, 0, len(t_src)-1)

    out = np.zeros((len(t_dst), voices), float)
    for k, i in enumerate(idx):
        ch = chords_hz[i]
        for v in range(voices):
            out[k, v] = float(ch[v] if v < len(ch) else ch[-1])
    return out, t_dst
