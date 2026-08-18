# model/

Put the trained model here:

```
backend/model/smile_classifier.keras
```

You get this file by running `training/smile_classifier_training.ipynb` in Google
Colab (the last cell downloads it). The backend loads it automatically on startup.

The model expects **raw 0–255 RGB images resized to 224×224** — preprocessing
(scaling to [-1, 1]) is baked into the model, so the backend must not scale pixels
again. See `services/model_service.py`.
