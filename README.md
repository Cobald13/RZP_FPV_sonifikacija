# FPV Telemetry Sonification

Ta repozitorij vsebuje prototipni sistem za **sonifikacijo telemetrije FPV drona**. Sistem podatke iz Betaflight Blackbox zapisov pretvori v zvočne parametre in jih uporabi za sintezo zvoka v okolju **Pure Data** ali za izvoz v **MIDI** zapis.

Projekt je bil razvit kot raziskovalno-razvojni prototip in je namenjen predvsem **offline uporabi** (CSV → zvok), z možnostjo realnočasovnega prenosa parametrov prek OSC.

---

## Funkcionalnosti

- uvoz Blackbox CSV datotek (Betaflight)
- obdelava IMU in ESC telemetrije (filtriranje, normalizacija, stabilizacija)
- preslikava podatkov v zvočne parametre:
  - višina tona (pitch)
  - amplituda
  - timbre
  - panorama
  - vibrato
- kvantizacija višin v izbrani tonaliteti
- generiranje akordov (triade / septakordi)
- izvoz v MIDI (dual-rate ali segmentacijski način)
- prenos parametrov prek OSC v Pure Data
- možnost snemanja izhoda iz Pure Data

---

## Struktura repozitorija

