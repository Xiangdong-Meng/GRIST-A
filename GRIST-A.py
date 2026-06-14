import cv2
import numpy as np
import random
import os
import glob

# ========== CONFIGURATION (modify as needed) ==========
input_folder = "./input"        # folder containing original images
output_folder = "./output"      # folder where augmented images will be saved

random.seed(42)

num_row, num_col = 12, 12
num_flip_patches = 100
color_tol = 8                   # color tolerance per channel (set 0 for exact PNG colors)
min_comp_area = 20              # minimum area for equiaxed connected components
max_shift = 300                 # maximum random shift (±300)
num_output_images = 6           # number of augmented images per input image

# BGR colors (OpenCV uses BGR)
BGR_CRACK = (0, 0, 255)         # crack -> red
BGR_EQUI = (0, 255, 0)          # equiaxed -> green
BGR_UNMELT = (255, 0, 0)        # unmelt -> blue
BGR_YSZ = (255, 255, 255)       # YSZ -> white
# =======================================================

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Collect all image files from input folder
image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
input_paths = []
for ext in image_extensions:
    input_paths.extend(glob.glob(os.path.join(input_folder, ext)))

print(f"Found {len(input_paths)} images to process")


# Generate binary mask from BGR color using inRange + tolerance
def mask_by_bgr(img_bgr, color_bgr, tol=0):
    c = np.array(color_bgr, dtype=np.int16)
    lower = np.clip(c - tol, 0, 255).astype(np.uint8)
    upper = np.clip(c + tol, 0, 255).astype(np.uint8)
    m = cv2.inRange(img_bgr, lower, upper)   # 0 or 255
    return (m > 0).astype(np.uint8)


# Rotate a region by 0/90/180/270 degrees
def rotate_region(region_img, region_mask, angle):
    if angle == 0:
        return region_img.copy(), region_mask.copy()
    if angle == 90:
        ri = cv2.rotate(region_img, cv2.ROTATE_90_CLOCKWISE)
        rm = cv2.rotate(region_mask, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        ri = cv2.rotate(region_img, cv2.ROTATE_180)
        rm = cv2.rotate(region_mask, cv2.ROTATE_180)
    elif angle == 270:
        ri = cv2.rotate(region_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rm = cv2.rotate(region_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("angle must be 0/90/180/270")
    return ri, (rm > 0).astype(np.uint8)


# Place a region at a random shift (±max_shift) from its original position,
# ensuring no overlap with previously placed equiaxed/unmelt regions.
def try_place_region_random(final_img, region_img, region_mask, orig_x, orig_y, max_shift, occupancy, attempts=500):
    h_r, w_r = region_mask.shape
    h, w = final_img.shape[:2]
    if h_r > h or w_r > w:
        return False
    for _ in range(attempts):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        nx = orig_x + dx
        ny = orig_y + dy
        nx = max(0, min(nx, w - w_r))
        ny = max(0, min(ny, h - h_r))
        occ_sub = occupancy[ny:ny + h_r, nx:nx + w_r]
        if np.any((occ_sub == 1) & (region_mask == 1)):
            continue
        # Place the region
        mask_bool = region_mask.astype(bool)
        tgt = final_img[ny:ny + h_r, nx:nx + w_r]
        tgt[mask_bool] = region_img[mask_bool]
        final_img[ny:ny + h_r, nx:nx + w_r] = tgt
        occupancy[ny:ny + h_r, nx:nx + w_r][mask_bool] = 1
        return True
    # Fallback: grid search
    step = max(1, min(16, max(1, min(h_r, w_r) // 4)))
    for yy in range(0, h - h_r + 1, step):
        for xx in range(0, w - w_r + 1, step):
            occ_sub = occupancy[yy:yy + h_r, xx:xx + w_r]
            if np.any((occ_sub == 1) & (region_mask == 1)):
                continue
            mask_bool = region_mask.astype(bool)
            tgt = final_img[yy:yy + h_r, xx:xx + w_r]
            tgt[mask_bool] = region_img[mask_bool]
            final_img[yy:yy + h_r, xx:xx + w_r] = tgt
            occupancy[yy:yy + h_r, xx:xx + w_r][mask_bool] = 1
            return True
    return False


# Variant that enforces a minimum shift in at least one axis (used for unmelt)
def try_place_region_random_with_min_shift(final_img, region_img, region_mask, orig_x, orig_y, max_shift, min_shift,
                                           occupancy, attempts=500):
    h_r, w_r = region_mask.shape
    h, w = final_img.shape[:2]
    for _ in range(attempts):
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        if abs(dx) < min_shift and abs(dy) < min_shift:
            continue
        nx = max(0, min(orig_x + dx, w - w_r))
        ny = max(0, min(orig_y + dy, h - h_r))
        occ_sub = occupancy[ny:ny + h_r, nx:nx + w_r]
        if np.any((occ_sub == 1) & (region_mask == 1)):
            continue
        mask_bool = region_mask.astype(bool)
        tgt = final_img[ny:ny + h_r, nx:nx + w_r]
        tgt[mask_bool] = region_img[mask_bool]
        final_img[ny:ny + h_r, nx:nx + w_r] = tgt
        occupancy[ny:ny + h_r, nx:nx + w_r][mask_bool] = 1
        return True
    return False


# Main processing function for a single image
def process_image(input_path, output_path_base, num_outputs=4):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Unable to read input image, skipping: {input_path}")
        return

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    h, w = img.shape[:2]

    # Generate masks (0/1)
    crack_mask = mask_by_bgr(img, BGR_CRACK, color_tol)
    equiaxed_mask = mask_by_bgr(img, BGR_EQUI, color_tol)
    unmelt_mask = mask_by_bgr(img, BGR_UNMELT, color_tol)
    ysz_mask = mask_by_bgr(img, BGR_YSZ, color_tol)

    # Build background: keep only crack + YSZ (replace equiaxed/unmelt with YSZ)
    background = img.copy()
    background[equiaxed_mask == 1] = BGR_YSZ
    background[unmelt_mask == 1] = BGR_YSZ

    # Extract equiaxed connected components (each becomes a separate movable patch)
    equiaxed_u8 = (equiaxed_mask * 255).astype(np.uint8)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(equiaxed_u8, connectivity=8)
    equiaxed_regions = []
    for lab in range(1, n_labels):
        x, y, w_c, h_c, area = stats[lab]
        if area < min_comp_area:
            continue
        comp_mask = (labels[y:y + h_c, x:x + w_c] == lab).astype(np.uint8)
        src_crop = img[y:y + h_c, x:x + w_c]
        region_img = np.zeros((h_c, w_c, 3), dtype=np.uint8)
        region_img[comp_mask == 1] = src_crop[comp_mask == 1]
        equiaxed_regions.append({
            "img": region_img,
            "mask": comp_mask,
            "orig_x": x,
            "orig_y": y,
            "w": w_c,
            "h": h_c,
            "area": area
        })

    # Extract the whole unmelt region as a single patch (no splitting)
    ys, xs = np.where(unmelt_mask == 1)
    unmelt_region_img = None
    unmelt_region_mask = None
    unmelt_orig_x = None
    unmelt_orig_y = None
    if len(xs) > 0:
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        unmelt_orig_x, unmelt_orig_y = min_x, min_y
        mask_crop = unmelt_mask[min_y:max_y + 1, min_x:max_x + 1].astype(np.uint8)
        h_r, w_r = mask_crop.shape
        src_crop = img[min_y:max_y + 1, min_x:max_x + 1]
        unmelt_region_img = np.zeros((h_r, w_r, 3), dtype=np.uint8)
        unmelt_region_img[mask_crop == 1] = src_crop[mask_crop == 1]
        unmelt_region_mask = mask_crop

    # Generate multiple augmented versions
    for i in range(num_outputs):
        random.seed(42 + i)   # ensure reproducibility per output

        # ---- Step 1: divide background into a grid and randomly flip some tiles ----
        patches = []
        coords = []
        patch_h = h // num_row
        patch_w = w // num_col
        for i_row in range(num_row):
            y0 = i_row * patch_h
            y1 = (i_row + 1) * patch_h if i_row < num_row - 1 else h
            for j_col in range(num_col):
                x0 = j_col * patch_w
                x1 = (j_col + 1) * patch_w if j_col < num_col - 1 else w
                patches.append(background[y0:y1, x0:x1].copy())
                coords.append((y0, y1, x0, x1))

        num_patches = len(patches)
        flip_n = min(num_flip_patches, num_patches)
        flip_idxs = random.sample(range(num_patches), flip_n)
        for idx in flip_idxs:
            patches[idx] = cv2.flip(patches[idx], 1)   # horizontal flip

        # Reassemble the transformed background
        recombined = np.zeros_like(background)
        for k, (y0, y1, x0, x1) in enumerate(coords):
            recombined[y0:y1, x0:x1] = patches[k]

        # ---- Step 2: place equiaxed and unmelt regions onto the transformed background ----
        occupancy = np.zeros((h, w), dtype=np.uint8)   # marks occupied pixels
        final_img = recombined.copy()

        # Place equiaxed patches (largest first to reduce placement conflicts)
        equiaxed_regions_sorted = sorted(equiaxed_regions, key=lambda x: -x["area"])
        angles = [0, 90, 180, 270]
        for reg in equiaxed_regions_sorted:
            angle = random.choice(angles)
            rimg, rmask = rotate_region(reg["img"], reg["mask"], angle)
            placed = try_place_region_random(final_img, rimg, rmask, reg["orig_x"], reg["orig_y"], max_shift, occupancy,
                                             attempts=500)
            if not placed:
                # fallback: place at original position (clipped to image boundaries)
                h_r, w_r = rmask.shape
                nx = max(0, min(reg["orig_x"], w - w_r))
                ny = max(0, min(reg["orig_y"], h - h_r))
                mask_bool = rmask.astype(bool)
                tgt = final_img[ny:ny + h_r, nx:nx + w_r]
                tgt[mask_bool] = rimg[mask_bool]
                final_img[ny:ny + h_r, nx:nx + w_r] = tgt
                occupancy[ny:ny + h_r, nx:nx + w_r][mask_bool] = 1

        # Place unmelt patch as a whole, with a non‑zero rotation and a minimum shift
        if unmelt_region_img is not None:
            angles_nonzero = [90, 180, 270]
            angle = random.choice(angles_nonzero)
            rimg_u, rmask_u = rotate_region(unmelt_region_img, unmelt_region_mask, angle)

            success_unmelt = try_place_region_random_with_min_shift(
                final_img, rimg_u, rmask_u, unmelt_orig_x, unmelt_orig_y,
                max_shift=max_shift, min_shift=50, occupancy=occupancy, attempts=600
            )

            if not success_unmelt:
                h_r, w_r = rmask_u.shape
                nx = max(0, min(unmelt_orig_x, w - w_r))
                ny = max(0, min(unmelt_orig_y, h - h_r))
                mask_bool = rmask_u.astype(bool)
                tgt = final_img[ny:ny + h_r, nx:nx + w_r]
                tgt[mask_bool] = rimg_u[mask_bool]
                final_img[ny:ny + h_r, nx:nx + w_r] = tgt
                occupancy[ny:ny + h_r, nx:nx + w_r][mask_bool] = 1

        # ---- Step 3: save the augmented image ----
        output_path = os.path.join(output_path_base, f"{base_name}_{i + 1}.png")
        cv2.imwrite(output_path, final_img)
        print(f"[DONE] Augmented image saved to: {output_path}")


# Process every image in the input folder
for input_path in input_paths:
    print(f"Processing image: {input_path}")
    process_image(input_path, output_folder, num_output_images)

print("All images processed successfully!")