# Star Identification - Algorithm Reference

This document provides a quick reference for each algorithm used in the star identification system, listing inputs, outputs, and purpose.

## Core Detection Algorithms

### 1. `detect_stars()`
**Purpose:** Detect star-like objects in astronomical images using computer vision

**Input:**
- `img`: OpenCV image array (BGR or grayscale)
- `gaussian_blur`: Blur kernel size (default: 3)
- `threshold_value`: Binary threshold (default: 70) 
- `min_area`: Minimum star area in pixels (default: 1)
- `max_area`: Maximum star area in pixels (default: 800)

**Output:**
- List of tuples: `[(x, y, radius, brightness), ...]`
- Each tuple represents one detected star

**Algorithm:**
1. Convert to grayscale
2. Apply Gaussian blur for noise reduction
3. Binary threshold to isolate bright objects
4. Find contours of bright regions
5. Filter by area to remove noise and artifacts
6. Extract centroid and brightness for each valid star

---

## Astrometric Calibration Algorithms

### 2. `solve_image_astrometry()`
**Purpose:** Determine sky coordinates and field of view using Astrometry.net API

**Input:**
- `image_path`: Path to temporary image file (string)

**Output:**
- `ra`: Right Ascension of image center (degrees) or None
- `dec`: Declination of image center (degrees) or None  
- `radius`: Field of view radius (degrees) or None

**Algorithm:**
1. Upload image to Astrometry.net service
2. Service matches star patterns to catalog
3. Returns World Coordinate System (WCS) solution
4. Extract image center coordinates and field size

### 3. `estimate_coordinates_from_filename()`
**Purpose:** Fallback coordinate estimation from filename patterns

**Input:**
- `filename`: Image filename (string)

**Output:**
- `ra`: Estimated Right Ascension (degrees) or None
- `dec`: Estimated Declination (degrees) or None
- `radius`: Estimated field radius (degrees) or None

**Algorithm:**
1. Convert filename to lowercase
2. Search for known astronomical object names
3. Return hardcoded coordinates for matched objects
4. Common patterns: 'orion', 'm42', 'pleiades', etc.

---

## Catalog Management Algorithms

### 4. `parse_ybc5_line()`
**Purpose:** Parse single line from Yale Bright Star Catalog fixed-width format

**Input:**
- `line`: Single text line from YBC5 catalog (string, 147+ characters)

**Output:**
- Dictionary with star data or None if parsing fails:
```python
{
    'hr': int,           # Harvard Revised number
    'name': str,         # Star name
    'ra': float,         # Right Ascension (degrees)
    'dec': float,        # Declination (degrees) 
    'magnitude': float,  # Visual magnitude
    'spectral_type': str # Spectral classification
}
```

**Algorithm:**
1. Extract fixed-width fields from specific character positions
2. Convert RA from hours/minutes/seconds to decimal degrees
3. Convert Dec from degrees/minutes/seconds to decimal degrees
4. Handle missing or invalid data gracefully

### 5. `load_ybc5_catalog()`
**Purpose:** Load entire Yale Bright Star Catalog from file

**Input:**
- `ybc5_path`: Path to YBC5 catalog file (default: 'data/YBC5')

**Output:**
- List of star dictionaries (same format as `parse_ybc5_line`)

**Algorithm:**
1. Open catalog file with Latin-1 encoding
2. Parse each line using `parse_ybc5_line()`
3. Collect valid star entries into list
4. Skip corrupted or incomplete lines

### 6. `find_stars_in_region()`
**Purpose:** Query catalog stars within specified sky region

**Input:**
- `ra`: Center Right Ascension (degrees)
- `dec`: Center Declination (degrees)
- `radius`: Search radius (degrees)

**Output:**
- Pandas DataFrame with columns: ['hr', 'name', 'ra', 'dec', 'magnitude', 'spectral_type']
- Sorted by magnitude (brightest first)

**Algorithm:**
1. Execute SQL box query on star database
2. Select stars within rectangular bounds: `ra ± radius`, `dec ± radius`
3. Sort results by visual magnitude
4. Return as structured DataFrame

---

## Coordinate Transformation Algorithms

### 7. Pixel-to-Sky Coordinate Conversion
**Purpose:** Convert image pixel coordinates to celestial coordinates

**Input:**
- `star_x`, `star_y`: Pixel coordinates
- `img_width`, `img_height`: Image dimensions
- `ra_center`, `dec_center`: Image center sky coordinates
- `radius`: Field of view radius

**Output:**
- `star_ra`: Star Right Ascension (degrees)
- `star_dec`: Star Declination (degrees)

**Algorithm:**
1. Calculate pixel scale: `(radius × 2) / min(width, height)` deg/pixel
2. Find pixel offset from image center
3. Convert pixel offset to angular offset
4. Apply spherical coordinate correction: `RA_offset / cos(Dec)`
5. Add offsets to image center coordinates

---

## Star Matching Algorithms  

### 8. Greedy Nearest-Neighbor Matching
**Purpose:** Match detected stars with catalog stars using position

**Input:**
- `detected_with_coords`: List of detected stars with sky coordinates
- `catalog_stars`: DataFrame of catalog stars in region
- `distance_threshold`: Maximum matching distance (default: 0.8°)

**Output:**
- `matches`: List of match dictionaries:
```python
{
    'x': float,              # Pixel coordinates
    'y': float,
    'name': str,             # Star name from catalog
    'hr': int,               # Harvard Revised number
    'magnitude': float,      # Visual magnitude
    'spectral_type': str,    # Spectral classification
    'distance': float        # Match distance (degrees)
}
```

**Algorithm:**
1. For each catalog star (sorted by brightness):
   - Find closest detected star not yet matched
   - Calculate Euclidean distance in sky coordinates
   - If distance < threshold, create match
   - Remove matched detected star from available pool
2. Continue until all catalog stars processed
3. One-to-one matching ensures no duplicates

---

## Image Annotation Algorithms

### 9. `create_detection_only_image()`
**Purpose:** Create annotated image showing only detected stars (no identification)

**Input:**
- `img`: Original image array
- `detected_df`: DataFrame of detected stars

**Output:**
- `annotated_img`: OpenCV image with red circles marking detected stars

**Algorithm:**
1. Create matplotlib figure with original image
2. Overlay red scatter points at detected star positions
3. Add title indicating number of detections
4. Convert plot to OpenCV image format

### 10. `identify_and_annotate_stars()` (Main Pipeline)
**Purpose:** Complete star identification pipeline with matplotlib visualization

**Input:**
- `img`: Input image array
- `filename`: Image filename for coordinate estimation
- `threshold_value`: Detection threshold (default: 120)

**Output:**
- `annotated_img`: Fully annotated image with labeled stars
- `result_info`: Dictionary with identification statistics and star data

**Algorithm:**
1. Run `detect_stars()` on input image
2. Attempt astrometric calibration via API and filename
3. Load catalog stars in image region
4. Transform detected star coordinates to sky coordinates  
5. Match detected stars with catalog using greedy algorithm
6. Create matplotlib visualization with:
   - Red circles for unidentified detections
   - Green circles with labels for identified stars
7. Generate comprehensive result statistics

### 11. `create_annotated_image_cv2()`
**Purpose:** Fast OpenCV-based image annotation (alternative to matplotlib)

**Input:**
- `img`: Original image array
- `detected_df`: DataFrame of detected stars
- `matches`: List of star matches
- `solve_method`: Method used for coordinate solving

**Output:**
- `annotated_img`: OpenCV image with annotations

**Algorithm:**
1. Copy original image
2. Draw red circles for all detected stars
3. Draw green circles for identified stars
4. Add text labels with star names and magnitudes
5. Draw arrows connecting labels to stars
6. Add title text with identification summary
7. Handle text positioning to avoid screen edges

---

## Utility Algorithms

### 12. `allowed_file()`
**Purpose:** Validate uploaded file extensions

**Input:**
- `filename`: Uploaded filename (string)

**Output:**
- `bool`: True if file extension is allowed

**Algorithm:**
1. Check if filename contains '.'
2. Extract file extension (lowercase)
3. Compare against allowed set: {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

### 13. Database Initialization (`create_and_populate()`)
**Purpose:** Initialize SQLite database with star catalog

**Input:**
- `ybc5_path`: Path to catalog file

**Output:**
- Populated SQLite database with star table

**Algorithm:**
1. Drop existing star table if present
2. Create new table with proper schema
3. Load all stars using `load_ybc5_catalog()`
4. Insert star records with duplicate handling
5. Commit transaction to database

---

## Performance Metrics

### Typical Algorithm Performance:
- **Star Detection:** 20-150 stars per image
- **Catalog Query:** ~100-500 stars per region  
- **Matching Success:** 5-15% identification rate
- **Processing Time:** 2-30 seconds total
- **Position Accuracy:** 10-60 arcseconds

### Computational Complexity:
- **Detection:** O(pixels) - linear in image size
- **Catalog Query:** O(log n) - database index lookup
- **Matching:** O(m × n) - m detected × n catalog stars
- **Annotation:** O(matches) - linear in number of matches