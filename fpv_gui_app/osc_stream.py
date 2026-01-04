# osc_stream.py
from __future__ import annotations
import time
import threading
import numpy as np
from pythonosc.udp_client import SimpleUDPClient
from fpv_core import chords_to_fps, resample_to_fps

class OscStreamer:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = False

        # REC support
        self._client: SimpleUDPClient | None = None
        self._rec_enabled: bool = False

    @property
    def running(self) -> bool:
        return self._running

    def stop(self):
        if self._running:
            self._stop.set()

            # takoj pošlji rec stop (če je možno)
            try:
                if self._rec_enabled and self._client is not None:
                    self._client.send_message("/rec", ["stop"])
            except Exception:
                pass

            if self._thread is not None:
                self._thread.join(timeout=1.0)

        self._running = False
        self._rec_enabled = False
        self._client = None

    def start(
        self,
        res: dict,
        fps: int,
        host: str,
        port: int,
        voices: int = 3,
        on_log=None,
        rec_path: str | None = None,
    ):
        if self._running:
            return
        self._stop.clear()

        def log(*args):
            if on_log:
                on_log(" ".join(map(str, args)))

        def run():
            try:
                client = SimpleUDPClient(host, port)
                t_sec = res["t_sec"]

                # --- REC start (optional) ---
                rec_enabled = bool(rec_path)
                rec_path_norm = None
                if rec_enabled:
                    rec_path_norm = str(rec_path).replace("\\", "/")
                    client.send_message("/rec", ["start", rec_path_norm])
                    log("REC start:", rec_path_norm)

                # --- precompute / resample ---
                freqs_f, _ = chords_to_fps(res["chords_hz"], t_sec, fps, voices=voices)
                amp_f, _   = resample_to_fps(res["amp"],    t_sec, fps)
                timb_f, _  = resample_to_fps(res["timbre"], t_sec, fps)
                pan_f, _   = resample_to_fps(res["pan"],    t_sec, fps)
                vibr_f, _  = resample_to_fps(res["vibr"],   t_sec, fps)

                # --- shaping / clamps ---
                amp_f = np.nan_to_num(amp_f, nan=0.0)
                amp_f = np.clip(amp_f, 0.0, 1.0)
                amp_f = 0.12 + 0.88 * (amp_f ** 0.6)

                timb_f = np.nan_to_num(timb_f, nan=0.0)
                timb_f = np.clip(timb_f, 0.0, 1.0)
                timb_f = 0.35 + 0.65 * timb_f

                pan_f  = np.clip(np.nan_to_num(pan_f,  nan=0.5), 0.0, 1.0)
                vibr_f = np.clip(np.nan_to_num(vibr_f, nan=0.0), 0.0, 1.0)

                n_frames = int(len(amp_f))
                log(f"OSC start → {host}:{port} @ {fps} fps, frames={n_frames}")
                if n_frames > 0:
                    # varno: izpiši do 3 glasove
                    vshow = min(voices, freqs_f.shape[1], 3) if freqs_f.ndim == 2 else 0
                    if vshow > 0:
                        log("first freqs:", freqs_f[0, :vshow], "amp", float(amp_f[0]),
                            "timbre", float(timb_f[0]), "pan", float(pan_f[0]), "vibr", float(vibr_f[0]))

                # --- streaming loop (frame-locked) ---
                t0 = time.perf_counter()
                for i in range(n_frames):
                    if self._stop.is_set():
                        break

                    # varno pobiranje f0,f1,f2 (tudi če voices < 3)
                    if freqs_f.ndim == 2 and freqs_f.shape[1] > 0:
                        f0 = float(freqs_f[i, 0])
                        f1 = float(freqs_f[i, 1]) if freqs_f.shape[1] > 1 else f0
                        f2 = float(freqs_f[i, 2]) if freqs_f.shape[1] > 2 else f1
                    else:
                        f0 = f1 = f2 = 0.0

                    msg = [
                        f0, f1, f2,
                        float(amp_f[i]),
                        float(timb_f[i]),
                        float(pan_f[i]),
                        float(vibr_f[i])
                    ]
                    client.send_message("/fpv", msg)

                    target = t0 + (i + 1) / float(fps)
                    dt = target - time.perf_counter()
                    if dt > 0:
                        time.sleep(dt)

                log("OSC stopped.")

            except Exception as e:
                log("OSC thread exception:", repr(e))

            finally:
                # --- REC stop (optional) ---
                try:
                    if rec_path:
                        client.send_message("/rec", ["stop"])
                        log("REC stop.")
                except Exception:
                    pass

                self._running = False
                self._rec_enabled = False
                self._client = None

        self._running = True
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
