# Dataset

Use a properly licensed line/word handwriting dataset such as IAM. Expected:

```text
dataset/
├── train/images/
├── train/labels.csv
├── validation/images/
├── validation/labels.csv
└── test/images/
    test/labels.csv
```

CSV columns: `filename,text`. Initial runtime support is English letters, digits and common punctuation.
