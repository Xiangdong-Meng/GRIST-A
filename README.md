# GRIST-A
GRIST-A: Official augmentation code for "Grid‑based Structure‑Preserving Augmentation for Predicting Elastic Modulus of TBCs". Random tile flips + patch repositioning (shift &amp; rotation) for colour‑coded micrographs. Preserves phase geometry while increasing diversity.

# Semantic Segmentation Data Augmentation – Patch Shuffling & Object Repositioning

This tool performs a **structured augmentation** for semantic segmentation datasets where different material phases are colour‑coded.  
It transforms the background via a random grid‑flipping operation and then re‑places the “equiaxed” and “unmelt” phase patches at shifted positions (with optional rotation), mimicking realistic microscopic variations while preserving the exact colour labels.

> **Key idea**  
> 1. Extract the **background** (crack + YSZ) and remove equiaxed/unmelt regions.  
> 2. Divide the background into a 12×12 grid and randomly flip some tiles horizontally.  
> 3. Re‑insert the extracted equiaxed/unmelt patches at locations that are randomly shifted (up to ±300 px) relative to their original positions.  
> 4. Generate several augmented images from each input.

## Dependencies

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

Install them with:

```bash
pip install opencv-python numpy
