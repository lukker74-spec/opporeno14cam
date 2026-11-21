# Reno14 FreshCam v7.1 SafeMemory (Fix Crash AI: ON)

- Risolve il crash (`OutOfMemoryError` / `BufferOverflowException`) durante lo scatto con **AI: ON**:
  - `android:largeHeap="true"` nel Manifest;
  - riciclo immediato (`recycle()`) di tutte le bitmap intermedie della raffica e dei tasselli;
  - supporto automatico ai tensori TFLite sia NHWC `[1, H, W, 3]` sia NCHW `[1, 3, H, W]` (come `SwinIR.tflite` e `QuickSRNetLarge.tflite`) con dimensionamento sicuro dei `ByteBuffer`.
