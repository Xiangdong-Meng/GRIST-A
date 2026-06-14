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

```

---

## How to Use

### 1. Prepare Input Images

Input images must be **colour-coded semantic segmentation maps**, where each class is represented by a unique BGR color.

Supported formats:

- .png
- .jpg
- .jpeg
- .bmp
- .tiff

---

### 2. Configure the Script

Edit parameters in `GRIST-A.py`:

- input_folder / output_folder
- num_output_images
- max_shift
- color_tol

---

## Default Color Coding (BGR)

| Class | BGR Value | Color |
|------|----------|------|
| Crack | (0,0,255) | Red |
| Equiaxed | (0,255,0) | Green |
| Unmelt | (255,0,0) | Blue |
| YSZ | (255,255,255) | White |

---

## Configurable Parameters

| Parameter | Default | Description |
|----------|--------|-------------|
| num_row | 12 | Grid rows |
| num_col | 12 | Grid cols |
| num_flip_patches | 100 | flipped tiles |
| color_tol | 8 | color tolerance |
| max_shift | 300 | displacement |
| num_output_images | 6 | outputs per image |

---

## Algorithm Overview

1. Mask Generation
2. Background Construction
3. Patch Extraction
4. Grid Flip Augmentation
5. Patch Repositioning
6. Output Composition

---

## Notes

- seed = 42 for reproducibility
- seed = 42 + i per image variation
- geometry preserved during augmentation

---

## License

This project is open‑source and available under the MIT License. Feel free to use and adapt it for your research.
```
