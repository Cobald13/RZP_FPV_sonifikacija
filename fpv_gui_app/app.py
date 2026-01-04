# app.py
from __future__ import annotations
import sys
import traceback
import numpy as np
from PySide6.QtCore import Qt, Signal

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QGroupBox
)
from PySide6.QtCore import Qt

import pyqtgraph as pg

from fpv_core import load_blackbox_csv, ComputeConfig, prepare_params_and_chords, NOTE_NAMES, SCALES
from fpv_core import chords_to_midi_dualrate, segments_to_midi, get_throttle_vector, segment_by_hysteresis
from osc_stream import OscStreamer

class MainWindow(QMainWindow):
    sig_log = Signal(str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FPV IMU → Audio (PySide6 + OSC)")
        self.resize(1200, 720)

        self.df = None
        self.fs = None
        self.res = None
        self.csv_path = None

        self.osc = OscStreamer()

        # -------- UI --------
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # Left controls
        left = QVBoxLayout()
        layout.addLayout(left, 0)

        # Right plots + log
        right = QVBoxLayout()
        layout.addLayout(right, 1)

        # --- File group ---
        file_box = QGroupBox("Data")
        file_l = QVBoxLayout(file_box)
        self.btn_open = QPushButton("Open CSV…")
        self.lbl_file = QLabel("No file loaded.")
        self.btn_compute = QPushButton("Compute / Preview")
        self.btn_compute.setEnabled(False)
        file_l.addWidget(self.btn_open)
        file_l.addWidget(self.lbl_file)
        file_l.addWidget(self.btn_compute)
        left.addWidget(file_box)

        # --- Mapping group ---
        map_box = QGroupBox("Mapping")
        map_l = QVBoxLayout(map_box)

        row1 = QHBoxLayout()
        self.cb_root = QComboBox(); self.cb_root.addItems(NOTE_NAMES); self.cb_root.setCurrentText("A")
        self.cb_scale = QComboBox(); self.cb_scale.addItems(list(SCALES.keys())); self.cb_scale.setCurrentText("minor")
        row1.addWidget(QLabel("Root")); row1.addWidget(self.cb_root)
        row1.addWidget(QLabel("Scale")); row1.addWidget(self.cb_scale)
        map_l.addLayout(row1)

        row2 = QHBoxLayout()
        self.cb_chord = QComboBox(); self.cb_chord.addItems(["triad","7","maj7","sus2","sus4"]); self.cb_chord.setCurrentText("7")
        self.sp_voices = QSpinBox(); self.sp_voices.setRange(1, 6); self.sp_voices.setValue(3)
        row2.addWidget(QLabel("Chord")); row2.addWidget(self.cb_chord)
        row2.addWidget(QLabel("Voices")); row2.addWidget(self.sp_voices)
        map_l.addLayout(row2)

        row3 = QHBoxLayout()
        self.sp_lp_acc = QDoubleSpinBox(); self.sp_lp_acc.setRange(5, 120); self.sp_lp_acc.setValue(30); self.sp_lp_acc.setSuffix(" Hz")
        self.sp_pitch_lp = QDoubleSpinBox(); self.sp_pitch_lp.setRange(0.2, 8.0); self.sp_pitch_lp.setValue(2.0); self.sp_pitch_lp.setSingleStep(0.1); self.sp_pitch_lp.setSuffix(" Hz")
        row3.addWidget(QLabel("Acc LP")); row3.addWidget(self.sp_lp_acc)
        row3.addWidget(QLabel("Pitch LP")); row3.addWidget(self.sp_pitch_lp)
        map_l.addLayout(row3)

        row4 = QHBoxLayout()
        self.sp_medk = QSpinBox(); self.sp_medk.setRange(1, 31); self.sp_medk.setValue(9)
        self.sp_hold = QSpinBox(); self.sp_hold.setRange(0, 500); self.sp_hold.setValue(120); self.sp_hold.setSuffix(" ms")
        self.sp_minstep = QSpinBox(); self.sp_minstep.setRange(0, 6); self.sp_minstep.setValue(2)
        row4.addWidget(QLabel("Median k")); row4.addWidget(self.sp_medk)
        row4.addWidget(QLabel("Hold")); row4.addWidget(self.sp_hold)
        row4.addWidget(QLabel("Min step")); row4.addWidget(self.sp_minstep)
        map_l.addLayout(row4)

        left.addWidget(map_box)

        # --- OSC group ---
        osc_box = QGroupBox("OSC")
        osc_l = QVBoxLayout(osc_box)

        rowo1 = QHBoxLayout()
        self.ed_host = QLineEdit("127.0.0.1")
        self.sp_port = QSpinBox(); self.sp_port.setRange(1, 65535); self.sp_port.setValue(57120)
        self.sp_fps = QSpinBox(); self.sp_fps.setRange(12, 120); self.sp_fps.setValue(25)
        rowo1.addWidget(QLabel("Host")); rowo1.addWidget(self.ed_host)
        rowo1.addWidget(QLabel("Port")); rowo1.addWidget(self.sp_port)
        rowo1.addWidget(QLabel("FPS")); rowo1.addWidget(self.sp_fps)
        osc_l.addLayout(rowo1)

        rowo2 = QHBoxLayout()
        self.btn_osc = QPushButton("Start OSC")
        self.btn_osc.setEnabled(False)
        rowo2.addWidget(self.btn_osc)
        osc_l.addLayout(rowo2)

        left.addWidget(osc_box)
        left.addStretch(1)

        # --- Plot (pyqtgraph) ---
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(title="Preview (first ~4000 samples)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve_pitch = self.plot.plot(pen=pg.mkPen("#00C2FF", width=2), name="pitch_midi")
        self.curve_amp   = self.plot.plot(pen=pg.mkPen("#FFB000", width=2), name="amp")
        self.curve_tim   = self.plot.plot(pen=pg.mkPen("#7CFF6B", width=2), name="timbre")
        right.addWidget(self.plot, 1)

        # --- Log ---
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        right.addWidget(self.log, 0)

        # Signal -> GUI log (thread-safe)
        self.sig_log.connect(self.log_msg)

        # -------- Signals --------
        self.btn_open.clicked.connect(self.on_open)
        self.btn_compute.clicked.connect(self.on_compute)
        self.btn_osc.clicked.connect(self.on_toggle_osc)

        # --- Export group ---
        export_box = QGroupBox("Export")
        ex = QVBoxLayout(export_box)

        row = QHBoxLayout()
        self.sp_bpm = QSpinBox(); self.sp_bpm.setRange(40, 200); self.sp_bpm.setValue(120)
        self.sp_notehz = QDoubleSpinBox(); self.sp_notehz.setRange(1, 40); self.sp_notehz.setValue(10.0)
        self.sp_cchz = QDoubleSpinBox(); self.sp_cchz.setRange(5, 200); self.sp_cchz.setValue(40.0)
        row.addWidget(QLabel("BPM")); row.addWidget(self.sp_bpm)
        row.addWidget(QLabel("Note Hz")); row.addWidget(self.sp_notehz)
        row.addWidget(QLabel("CC Hz")); row.addWidget(self.sp_cchz)
        ex.addLayout(row)

        self.btn_midi_dual = QPushButton("Export MIDI (dualrate)")
        self.btn_midi_seg  = QPushButton("Export MIDI (segments)")
        self.btn_midi_dual.setEnabled(False)
        self.btn_midi_seg.setEnabled(False)
        ex.addWidget(self.btn_midi_dual)
        ex.addWidget(self.btn_midi_seg)
        left.addWidget(export_box)

        self.btn_midi_dual.clicked.connect(self.on_export_midi_dual)
        self.btn_midi_seg.clicked.connect(self.on_export_midi_segments)

    def log_msg(self, s: str):
        self.log.appendPlainText(s)

    def on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Blackbox CSV", "", "CSV Files (*.csv);;All Files (*.*)")
        if not path:
            return
        try:
            df, fs = load_blackbox_csv(path)
            self.df, self.fs, self.csv_path = df, fs, path
            self.lbl_file.setText(f"Loaded: {path}\nfs≈{fs:.2f} Hz | rows={len(df)}")
            self.btn_compute.setEnabled(True)
            self.btn_osc.setEnabled(False)
            self.res = None
            self.log_msg(f"✅ CSV loaded: {path}")
        except Exception as e:
            self.log_msg("❌ Load failed: " + str(e))
            self.log_msg(traceback.format_exc())

    def current_cfg(self) -> ComputeConfig:
        return ComputeConfig(
            root=self.cb_root.currentText(),
            scale=self.cb_scale.currentText(),
            chord_type=self.cb_chord.currentText(),
            voices=int(self.sp_voices.value()),
            lp_cut=float(self.sp_lp_acc.value()),
            pitch_med_k=int(self.sp_medk.value()),
            pitch_lp_hz=float(self.sp_pitch_lp.value()),
            hold_ms=int(self.sp_hold.value()),
            min_step=int(self.sp_minstep.value()),
        )

    def on_compute(self):
        if self.df is None:
            return
        try:
            cfg = self.current_cfg()
            res = prepare_params_and_chords(self.df, self.fs, cfg)
            self.res = res

            t_sec = res["t_sec"]
            dur = float(t_sec[-1]) if len(t_sec) else 0.0

            voices = cfg.voices
            freqs0 = np.array([ch[:voices] for ch in res["chords_hz"]], dtype=float)

            self.log_msg(f"✅ Compute OK | duration={dur:.2f}s | voices={voices}")
            self.log_msg(f"freq min/max: {np.nanmin(freqs0):.2f} .. {np.nanmax(freqs0):.2f}")
            self.log_msg(f"quant unique: {len(np.unique(res['quant_midi']))} | changes: {int(np.sum(np.diff(res['quant_midi'])!=0))}")

            n = min(4000, len(res["pitch_midi"]))
            x = t_sec[:n]
            self.curve_pitch.setData(x, res["pitch_midi"][:n])
            self.curve_amp.setData(x, res["amp"][:n])
            self.curve_tim.setData(x, res["timbre"][:n])
            
            self.btn_midi_dual.setEnabled(True)
            self.btn_midi_seg.setEnabled(True)

            self.btn_osc.setEnabled(True)
        except Exception as e:
            self.log_msg("❌ Compute failed: " + str(e))
            self.log_msg(traceback.format_exc())
            
    def on_export_midi_dual(self):
        if self.res is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save MIDI (dualrate)", "drone_score_dualrate.mid", "MIDI (*.mid)")
        if not path:
            return
        try:
            bpm = int(self.sp_bpm.value())
            notehz = float(self.sp_notehz.value())
            cchz = float(self.sp_cchz.value())
            voices = int(self.sp_voices.value())

            mid = chords_to_midi_dualrate(
                self.res["chords_midi"], self.res["amp"], self.res["t_sec"],
                bpm=bpm, voices=voices, note_rate_hz=notehz, cc_rate_hz=cchz
            )
            mid.save(path)
            self.log_msg(f"💾 Saved MIDI dualrate: {path}")
        except Exception as e:
            self.log_msg("❌ MIDI dualrate export failed: " + str(e))

    def on_export_midi_segments(self):
        if self.res is None or self.df is None or self.fs is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save MIDI (segments)", "drone_score_segments.mid", "MIDI (*.mid)")
        if not path:
            return
        try:
            # minimal defaults (lahko kasneje damo UI kontrolnike)
            thr_lp = 3.0
            lo, hi = 0.35, 0.45
            min_len_s = 0.30
            nbins = 4

            thr = get_throttle_vector(self.df, self.fs, lp_cut=thr_lp)
            segs = segment_by_hysteresis(thr, self.fs, lo=lo, hi=hi, min_len_s=min_len_s, nbins=nbins)

            voices = int(self.sp_voices.value())
            mid = segments_to_midi(segs, self.res["chords_midi"], self.res["amp"], self.res["t_sec"], bpm=60, voices=voices)
            mid.save(path)
            self.log_msg(f"💾 Saved MIDI segments: {path} | segs={len(segs)}")
        except Exception as e:
            self.log_msg("❌ MIDI segments export failed: " + str(e))

    def on_toggle_osc(self):
        if self.res is None:
            return

        if not self.osc.running:
            host = self.ed_host.text().strip()
            port = int(self.sp_port.value())
            fps = int(self.sp_fps.value())
            voices = int(self.sp_voices.value())

            wav_path, _ = QFileDialog.getSaveFileName(
                self, "Save recording WAV", "take.wav", "WAV (*.wav)"
            )
            if not wav_path:
                return

            wav_path = wav_path.replace("\\", "/")

            self.osc.start(
                self.res, fps=fps, host=host, port=port, voices=voices,
                on_log=self.sig_log.emit,
                rec_path=wav_path
            )
            self.btn_osc.setText("Stop OSC")
        else:
            self.osc.stop()
            self.btn_osc.setText("Start OSC")
            self.sig_log.emit("⏹ OSC stop requested.")

    def closeEvent(self, event):
        try:
            self.osc.stop()
        except Exception:
            pass
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
