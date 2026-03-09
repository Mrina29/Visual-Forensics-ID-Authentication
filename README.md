# Visual Forensics for Identity Document Authentication
**Using Multi-Feature Analysis (Fonts, Textures, Seals & Holograms)**

### Project Overview
This repository contains the source code for an automated computer vision-based system that verifies the authenticity of identity documents offline, without relying on external databases. 

### Methodology (Multi-Feature ROI Extraction)
1. **Pre-processing:** Automatic document detection and perspective warping (OpenCV).
2. **Micro-Textures:** Local Binary Patterns (LBP) to analyze background guilloche patterns.
3. **Fonts & Typography:** LBP to detect pixelation and printer anomalies.
4. **Seals & Holograms:** ResNet18 CNN features to analyze complex holographic regions.
5. **Classification:** Random Forest Classifier outputting Genuine/Counterfeit with a confidence score.

### How to Run
```bash
pip install -r requirements.txt
python train.py
python -m streamlit run app.py
