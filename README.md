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


## How to Use

### 1\. Prepare Input Images

- All input images must be **colour images**\. Each semantic class needs to be painted with a unique BGR colour \(refer to the default colour mapping below\)\.

- **Supported image formats**: \`\.png\`, \`\.jpg\`, \`\.jpeg\`, \`\.bmp\`, \`\.tiff\`

### 2\. Adjust Configuration

Modify the configurable parameters at the top of `enhance_dataset.py`:

- `input_folder` / `output_folder`: Set your custom input and output file paths

- `num_output_images`: Number of augmented images generated for each single input image

- `max_shift`: Maximum pixel displacement for equiaxed and unmelt patch repositioning

- `color_tol`: Color tolerance for mask extraction \(compatible with JPEG compression artifacts\)

Detailed parameter explanations are listed in the **Configurable Parameters** section below\.

### 3\. Run the Script

```bash
python enhance_dataset.py
```

Augmented images will be saved to the specified `output_folder` with the naming pattern:`originalname_1.png`, `originalname_2.png`, \.\.\.

## Default Colour Coding \(BGR\)

You can modify these BGR colour values in the configuration section of the script\.

|Class|Colour \(B, G, R\)|Display Colour|
|---|---|---|
|Crack|\(0, 0, 255\)|Red|
|Equiaxed|\(0, 255, 0\)|Green|
|Unmelt|\(255, 0, 0\)|Blue|
|YSZ|\(255, 255, 255\)|White|

## Configurable Parameters

|Parameter|Default|Description|
|---|---|---|
|`num_row`, `num_col`|12, 12|Grid size for background tile splitting|
|`num_flip_patches`|100|Number of random tiles that are horizontally flipped for background augmentation|
|`color_tol`|8|Tolerance per BGR channel during mask generation; increase the value for lossy compressed images \(e\.g\., JPEG\)|
|`min_comp_area`|20|Minimum pixel area for valid equiaxed connected components; smaller components will be ignored|
|`max_shift`|300|Maximum random translation offset \(±max\_shift\) for patch repositioning|
|`num_output_images`|6|Number of augmented images generated for each original input image|
|`min_shift` \(for unmelt\)|50|Minimum displacement for unmelt patch repositioning to ensure sufficient data variation|

## Algorithm Outline

1. **Mask Generation**Extract binary masks for four semantic classes \(Crack, Equiaxed, Unmelt, YSZ\) based on the predefined BGR colour values\.

2. **Background Creation**Replace all equiaxed and unmelt pixel regions with YSZ white colour to generate a pure background template\.

3. **Patch Extraction**

    - Equiaxed: Each independent connected component is extracted as a separate patch

    - Unmelt: The entire unmelt region is treated as a single integral patch \(no segmentation\)

4. **Background Augmentation**Split the background image into uniform grid tiles, randomly flip partial tiles horizontally, and reassemble the tiles to generate a varied background\.

5. **Patch Repositioning**

    - Equiaxed patches: Rotate by 0°/90°/180°/270°, then place at a random offset \(±`max_shift`\) from the original position with overlap avoidance

    - Unmelt patch: Rotate only by 90°/180°/270° \(no 0° rotation\), with a minimum 50px displacement to prevent negligible changes

6. **Image Saving**Combine the augmented background and repositioned patches, then save the final composite image to the output directory\.

## Notes

- The script uses a fixed global random seed \(`42`\) for full reproducibility\. Each augmented image adopts an incremental seed \(`42 + i`\) to guarantee diverse output results\.

- If a patch fails to be placed without overlap after multiple attempts, it will fallback to its original position \(clipped to image boundaries\)\.

- The unmelt patch excludes 0° rotation by default to ensure obvious data augmentation effects\.

## License

This project is open\-source under the **MIT License**\. Feel free to use, modify and adapt the code for academic and research purposes\.

> （注：文档部分内容可能由 AI 生成）
