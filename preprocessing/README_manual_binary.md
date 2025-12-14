# Image Preprocessing Pipeline for 2PLSM Data

## Overview

This script implements a classical image preprocessing pipeline for Two-Photon Laser Scanning Microscopy (2PLSM) images, aiming to generate clean and standardized binary masks of central dendritic structures.

The pipeline is designed for **exploratory analysis and feature preprocessing**, rather than fully automated segmentation. Manual parameter tuning is recommended, as fully automatic segmentation may not perform reliably across all imaging conditions.

---

## Processing Logic

The preprocessing pipeline consists of the following steps:

1. **Load and normalize the input image**  
   
   - Input images are converted to grayscale and intensity-normalized.

2. **Crop the central region of interest (ROI)**  
   
   - Only the central portion of the image is retained to suppress peripheral background noise.

3. **Median filtering**  
   
   - Reduces photon noise while preserving thin dendritic boundaries.

4. **Adaptive local thresholding**  
   
   - Converts the smoothed image into an initial binary mask.

5. **Morphological smoothing**  
   
   - Morphological opening and closing operations remove small artifacts and fill gaps.

6. **Central connected-component selection**  
   
   - Only the connected white component containing the image center is retained.

7. **Resize to a fixed resolution (250 × 250)**  
   
   - Ensures a consistent input size for downstream analysis.

Intermediate results are visualized at each step to facilitate debugging and parameter tuning.

---

## Important Limitations

⚠ **This pipeline is NOT fully automatic.**  
Its performance depends strongly on input image quality and parameter selection.

Specifically:

- Thresholding parameters (e.g. `block_size`, `offset`) must be manually tuned for different datasets or imaging conditions.
- Morphological kernel sizes may require adjustment depending on dendritic spine thickness.
- The pipeline does not learn parameters from data and therefore cannot adapt automatically.

As a result, segmentation quality may vary, and suboptimal results are expected without careful tuning.

---

## Input Requirements

To obtain reasonable results, the following conditions should be satisfied:

- **The target structure should be approximately centered in the image**  
  
  - The pipeline explicitly retains only the connected component at the image center.

- **Image orientation should be consistent**  
  
  - Images should be rotated or aligned beforehand if necessary.

- **Each image should contain a single dominant foreground structure**  
  
  - Multiple competing foreground objects may lead to incorrect component selection.

Failure to meet these conditions will significantly degrade segmentation performance.
