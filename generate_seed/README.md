# Uterus Segmentation Seed Generation

This project provides two modular pipelines to generate seed points for segmenting the **mouse uterus** from **T2***-weighted MRI slices**. The seeds can be used to initialize segmentation algorithms like watershed, region growing, or nnU-Net.

## 🧠 Purpose

Segmenting the mouse uterus from whole-body MR scans can be challenging due to variations in image contrast and abdominal signal. This toolkit offers two adaptive pipelines:

- **Frangi-based pipeline**: Ideal for images where the abdominal cavity contains significant signal from fat.
- **Morphology-based pipeline**: Best suited when the abdominal cavity's fat is suppressed, and uterus is a prominent blob.

---

## 📁 Project Structure

```
uterus-segmentation-seeding/
├── frangi_pipeline/
│   ├── enhancement.py
│   ├── seed_extraction.py
│   └── run_frangi_pipeline.py
├── morphology_pipeline/
│   └── morphology_based_seeding.py
├── utils/
│   └── image_loader.py
├── data/
│   └── [example MRI slices]
├── requirements.txt
└── README.md
```

---

## 🚀 How to Use

### 1. Install dependencies

### 2. Run the Frangi pipeline

```bash
python frangi_pipeline/run_frangi_pipeline.py
```

or

### 3. Run the Morphology-based pipeline

```bash
python morphology_pipeline/morphology_based_seeding.py
```

---

## 🧪 Input Requirements

- Format: `.png`, `.tif`, or `.jpg` slices  
- Should be grayscale or RGB/RGBA (auto-converted)  
- Example images can be placed in `data/`

---

## 🛠️ Output

Both scripts return **binary seed masks** or seed coordinates, and provide visualizations for verification.

---

## 🧭 When to Use Which?

| Scenario                               | Recommended Pipeline |
| -------------------------------------- | -------------------- |
| Abdominal cavity contains signal       | Frangi-based         |
| Abdominal cavity is mostly dark/empty  | Morphology-based     |
