# Reno14 FreshCam v13.0 (Motore Computazionale Locale + Zero Freeze + Ghiera Fluida + Cloud Radiale)

- **Nuovo motore locale di fotografia computazionale (zero TFLite a tasselli):** piramide a 3 scale su sola luminanza con Noise-Gate (zero rumore su cielo/pelle), Local Tone Mapping (recupero ombre + protezione alte luci) e chroma denoise veloce.
- **Risolti i blocchi dell'app:** rimosso NNAPI dal Main Thread, aggiunto debounce sul binding CameraX, EV e Flash aggiornati live senza riavviare la fotocamera, blocco anti-doppio-scatto e riduzione del 60% del picco RAM.
- **Ghiera zoom fluida + Pinch-to-Zoom:** corretto il bug di `pointerInput` che bloccava la rotazione; ora gira istantaneamente fino a 120x e funziona anche con le due dita sul mirino.
- **Cloud AI sostituita e corretta:** passata a `Swin2SR Classical x2` (addestrato su foto pulite) con fusione di sola luminanza e maschera radiale morbida (niente riquadro rettangolare, colori originali preservati).
