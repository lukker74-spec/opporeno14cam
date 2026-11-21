# Reno14 FreshCam v3.0 (Rete Neurale ESRGAN Reale)

Integra TensorFlow Lite (`org.tensorflow:tensorflow-lite:2.16.1`) e scarica automaticamente il modello neurale `esrgan.tflite` in fase di build su GitHub Actions. Quando `SR: ON` è attivo, esegue inferenza neurale reale ESRGAN a tasselli sull'area centrale dell'immagine salvata.
