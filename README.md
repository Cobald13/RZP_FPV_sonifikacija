# Pretvorba podatkov iz IMU senzorjev FPV drona vparametre sinteze

Ta projekt predstavlja sistem za **sonifikacijo telemetrijskih podatkov FPV drona**, kjer se podatki iz Betaflight Blackbox zapisov pretvorijo v zvočne parametre in uporabijo za sintezo zvoka ali izvoz v MIDI.

Sistem je zasnovan kot cevovod:

**Blackbox CSV → obdelava signalov → preslikava v zvočne parametre → sinteza (Pure Data) ali MIDI**

---

## Funkcionalnosti

- uvoz Betaflight Blackbox CSV datotek
- obdelava IMU in ESC telemetrije (filtriranje, normalizacija, stabilizacija)
- preslikava podatkov v zvočne parametre:
  - višina (pitch)
  - amplituda
  - timbre
  - panorama
  - vibrato
- kvantizacija višin v izbrani tonaliteti
- generiranje večglasnih akordov
- **realnočasovni OSC prenos v Pure Data**
- snemanje zvočnega izhoda iz Pure Data
- izvoz v MIDI (dual-rate ali segmentacijski način)
- grafični uporabniški vmesnik (Python + PySide6)

---

## Zahteve

### Programska oprema

- **Python 3.9+**
- **Pure Data (Pd Vanilla ali Pd Extended)**  
  https://puredata.info/

### Python knjižnice

Seznam je v `requirements.txt`:
- numpy
- pandas
- scipy
- mido
- python-osc
- PySide6
- pyqtgraph

---

## Namestitev

### Kloniranje repozitorija, venv, requirements

```bash
git clone https://github.com/USERNAME/fpv-telemetry-sonification.git
cd fpv-telemetry-sonification

python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Zagon aplikacije
```bash
python app.py
```

Odpre se grafični vmesnik, kjer lahko:
- naložiš CSV datoteko
- nastaviš tonaliteto, tip akorda in parametre filtriranja
- vidiš predogled signalov
- izbereš izhod (OSC ali MIDI)

### Zagon Pure Data patcha (za OSC)
- Odpri Pure Data
- Naloži patch:
```bash
puredataSynth.pd
```
- Preveri:
  - OSC port (privzeto: 57120)
  - audio output nastavitve (Zavihek Media -> DSP ON)

---

Aplikacija omogoča tako izvoz signala v MIDI kot tudi realnočasovno sintezo v PureData

### Realnočasovna sinteza (OSC)
V Python aplikaciji:
- naloži csv Dals.csv
- klikni Compute / Preview
- klikni Start OSC
- izberi pot za snemanje .wav

Parametri se pošiljajo prek OSC sporočila:
```bash
/fpv [f0, f1, f2, A, T, P, V]
```

### Izvoz v MIDI
Aplikacija omogoča:
- dual-rate MIDI (ločena hitrost za note in kontrolne dogodke)
- segmentacijski MIDI (segmentacija glede na throttle)

Izvoženo MIDI datoteko lahko odpreš v kateremkoli DAW (Ableton, Reaper, Logic, …)

V Python aplikaciji:
- naloži csv Dals.csv
- klikni Compute / Preview
- klikni Export MIDI (dualrate/segments)
