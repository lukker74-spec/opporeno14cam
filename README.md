# Reno14 FreshCam v18.0 (Vera AI Neurale Locale ONNX + NPU NNAPI + RAW-in-RAM → JPEG)

- **Vera AI Generativa / Super-Resolution 100% Locale Offline (`OnnxGenerativeEngine.kt`):**
  - Integra `onnxruntime-android:1.24.2` con due veri modelli neurali scaricati automaticamente in fase di build:
    1. **`4x-ClearRealityV1.onnx` (SPAN 4x, 1.7 MB, predefinito):** veloce e fotorealistico, pulisce la grana e ricostruisce texture naturali.
    2. **`realesr-general-x4v3.onnx` (Real-ESRGAN v3 4x, 4.9 MB):** ricostruzione più spinta per zoom estremi (20x–120x).
  - Prova l'accelerazione **Android NNAPI FP16** (per delegare alla NPU/DSP Dimensity quando supportata) con fallback automatico e trasparente a **CPU ARM 4-Thread**, mostrando backend e tempo reale nel banner di salvataggio.
- **Zero Cloud a pagamento:** Replicate rimosso completamente; resta solo l'opzione gratuita opzionale (Hugging Face con token gratuito `hf_...` o endpoint gratuito personalizzato) con avviso esplicito se non configurata.
- **Fix banner con `AI Lock`:** il blocco fuoco non nasconde più i messaggi di salvataggio, tempi AI o errori.
