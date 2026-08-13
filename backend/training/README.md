# Custom CRNN + BiLSTM + CTC

Use line-level image/transcription pairs. The runtime demo uses PP-OCRv5; this folder is the custom-model path for a project that must train its own HTR network.

```text
dataset/train/images/*.png
dataset/train/labels.csv
```

`labels.csv`: `filename,text`.

A production CTC trainer must pass padded labels and input/label lengths to `tf.nn.ctc_loss`. Do not train this architecture with isolated EMNIST character labels.
