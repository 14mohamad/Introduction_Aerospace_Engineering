# Star Identification Algorithm - Complete Guide

This document provides a detailed explanation of the star identification system, covering every step from image input to final annotated output.

## System Architecture

The system is split into two main components:

1. **`star_algorithm.py`** - Core algorithmic logic
2. **`flask_star_identifier.py`** - Web server wrapper

## Complete Algorithm Flow

### Step 1: Image Input & Preprocessing

**Input:** Astronomical image (PNG, JPG, JPEG, GIF, BMP)
**Location:** Flask endpoints (`/identify`, `/identify_fast`, `/identify_raw`)

```python
# Image validation and loading
file_bytes = np.frombuffer(file.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
```

**Process:**
- Validate file format against allowed extensions
- Convert uploaded file to OpenCV image format
- Handle corrupted or invalid image files

### Step 2: Star Detection (Computer Vision)

**Function:** `detect_stars()` in `star_algorithm.py`

**Algorithm:**
1. **Grayscale Conversion**
   ```python
   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   ```

2. **Noise Reduction**
   ```python
   blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
   ```
   - Gaussian blur with 3x3 kernel
   - Reduces noise while preserving star shapes

3. **Binary Thresholding**
   ```python
   _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
   ```
   - Default threshold: 120 (configurable)
   - Separates bright stars from dark background

4. **Contour Detection**
   ```python
   contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   ```
   - Finds connected bright regions
   - Each contour represents a potential star

5. **Star Filtering**
   ```python
   area = cv2.contourArea(cnt)
   if min_area < area < max_area:
   ```
   - Filter by area (1-800 pixels)
   - Removes noise (too small) and planets/artifacts (too large)

6. **Star Properties Extraction**
   ```python
   (x, y), radius = cv2.minEnclosingCircle(cnt)
   brightness = img_gray[int(y), int(x)]
   ```

**Output:** List of detected stars with (x, y, radius, brightness)

### Step 3: Astrometric Calibration (Plate Solving)

**Goal:** Convert pixel coordinates to sky coordinates (RA/Dec)

#### Method 1: Astrometry.net API
**Function:** `solve_image_astrometry()` in `star_algorithm.py`

```python
ast = AstrometryNet()
ast.api_key = 'nuawxvldlgovsbwo'
wcs_header = ast.solve_from_image(image_path)
```

**Process:**
1. Save image to temporary file
2. Upload to astrometry.net service
3. Service analyzes star patterns
4. Returns World Coordinate System (WCS) solution:
   - `CRVAL1` (RA of image center)
   - `CRVAL2` (Dec of image center)  
   - `RADIUS` (field of view)

#### Method 2: Filename Pattern Recognition
**Function:** `estimate_coordinates_from_filename()` in `star_algorithm.py`

**Fallback for common astronomical objects:**
```python
regions = {
    'orion': (85.0, -1.0, 15.0),      # RA, Dec, radius in degrees
    'm42': (83.8, -5.4, 2.0),
    'm31': (10.7, 41.3, 3.0),
    'pleiades': (56.9, 24.1, 2.0),
    # ... more regions
}
```

### Step 4: Star Catalog Loading

**Class:** `CompleteCatalog` in `star_algorithm.py`

#### Catalog: Yale Bright Star Catalog v5 (YBC5)
- **Source:** `data/YBC5` file
- **Contains:** ~9,000 brightest stars
- **Format:** Fixed-width ASCII

#### Parsing Process:
**Function:** `parse_ybc5_line()`

```python
# Extract data from fixed positions
hr = int(line[0:4].strip())           # Harvard Revised number
name = line[4:14].strip()             # Star name
ra_h = int(line[75:77])               # RA hours
ra_m = int(line[77:79])               # RA minutes  
ra_s = float(line[79:83])             # RA seconds
dec_d = int(line[84:86])              # Dec degrees
# ... convert to decimal degrees
```

#### Database Storage:
```sql
CREATE TABLE stars (
    hr INTEGER PRIMARY KEY,
    name TEXT,
    ra REAL,
    dec REAL, 
    magnitude REAL,
    spectral_type TEXT
)
```

#### Spatial Query:
**Function:** `find_stars_in_region()`

```python
cursor.execute('''
    SELECT * FROM stars 
    WHERE ra BETWEEN ? AND ? 
    AND dec BETWEEN ? AND ?
    ORDER BY magnitude
''', (ra - radius, ra + radius, dec - radius, dec + radius))
```

### Step 5: Coordinate Transformation

**Goal:** Convert detected star pixel positions to sky coordinates

#### Pixel Scale Calculation:
```python
pixel_scale_deg_per_pixel = (radius * 2) / min(width, height)
```

#### Coordinate Conversion:
```python
# Pixel offset from image center
x_pixels_from_center = star['x'] - width/2
y_pixels_from_center = -(star['y'] - height/2)  # Flip Y axis

# Convert to angular offsets
x_deg_offset = x_pixels_from_center * pixel_scale_deg_per_pixel
y_deg_offset = y_pixels_from_center * pixel_scale_deg_per_pixel

# Apply spherical coordinate correction
star_ra = ra + x_deg_offset / np.cos(np.radians(dec))
star_dec = dec + y_deg_offset
```

**Key Considerations:**
- Y-axis flip (image vs. sky coordinates)
- Spherical coordinate correction for RA
- Simple linear projection (adequate for small fields)

### Step 6: Star Matching Algorithm

**Goal:** Match detected stars with catalog stars

#### Algorithm: Greedy Nearest-Neighbor
```python
for _, cat_star in catalog_stars.iterrows():
    best_detected = None
    min_dist = float('inf')
    
    for det_star in detected_with_coords:
        # Euclidean distance in sky coordinates
        dist = np.sqrt((det_star['ra'] - cat_star['ra'])**2 + 
                      (det_star['dec'] - cat_star['dec'])**2)
        
        if dist < min_dist and dist < 0.8:  # 0.8 degree threshold
            min_dist = dist
            best_detected = det_star
    
    if best_detected is not None:
        # Create match and remove from pool
        matches.append(match_data)
        used_catalog_stars.add(cat_star['hr'])
        detected_with_coords.remove(best_detected)
```

**Matching Criteria:**
- Maximum separation: 0.8 degrees
- One-to-one matching (no duplicate assignments)
- Preference for brighter catalog stars (sorted by magnitude)

### Step 7: Image Annotation

**Functions:** 
- `identify_and_annotate_stars()` - Full pipeline with matplotlib
- `create_annotated_image_cv2()` - OpenCV-based annotation

#### Matplotlib Version (Main Algorithm):
```python
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Plot detected stars (red circles)
plt.scatter(detected_df['x'], detected_df['y'], s=50, c='red', 
            marker='o', edgecolors='white', alpha=0.5)

# Plot identified stars (green circles with labels)
for match in matches:
    plt.scatter(match['x'], match['y'], s=200, facecolors='none', 
               edgecolors='lime', linewidths=2)
    plt.annotate(f"{match['name']}\nHR{match['hr']} (mag {match['magnitude']:.1f})",
                xy=(match['x'], match['y']), xytext=(10, 10),
                textcoords='offset points', fontsize=8, color='yellow')
```

#### OpenCV Version (Fast Rendering):
```python
# Draw detected stars
cv2.circle(annotated_img, (x, y), 8, (0, 0, 255), 2)  # Red circle

# Draw identified stars  
cv2.circle(annotated_img, (x, y), 12, (0, 255, 0), 3)  # Green circle
cv2.putText(annotated_img, label, (text_x, text_y), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
```

### Step 8: Result Generation

**Output Data Structure:**
```python
result_info = {
    "stars_detected": len(detected_df),
    "stars_identified": len(matches), 
    "identification_rate": f"{len(matches)/len(detected_df)*100:.1f}%",
    "solve_method": solve_method,  # "astrometry.net", "filename pattern", "none"
    "image_center": {"ra": ra, "dec": dec},
    "field_of_view": radius * 2,
    "identified_stars": [
        {
            "name": match['name'],
            "hr": match['hr'], 
            "magnitude": match['magnitude'],
            "spectral_type": match['spectral_type'],
            "position": {"x": match['x'], "y": match['y']}
        } for match in sorted(matches, key=lambda x: x['magnitude'])
    ]
}
```

## Web Server Endpoints

### `/identify` - Full Processing with Base64 Image
- Returns JSON with base64-encoded annotated image
- Best for web applications with inline image display

### `/identify_fast` - Fast Processing with Image URL  
- Saves annotated image to uploads/ directory
- Returns JSON with image URL
- More efficient for repeated access

### `/identify_raw` - Direct Image Download
- Returns annotated image file directly
- Best for API integrations

## Performance Characteristics

### Typical Results:
- **Stars Detected:** 20-150 per image
- **Stars Identified:** 5-25 per image (5-15% identification rate)
- **Processing Time:** 2-30 seconds (depending on astrometry.net)
- **Position Accuracy:** 10-60 arcseconds

### Limiting Factors:
1. **Catalog Completeness:** Only ~9,000 brightest stars
2. **Field of View:** Optimized for 1-10 degree fields
3. **Image Quality:** Requires focused, well-exposed stars
4. **Coordinate Precision:** Simple linear projection model

## Error Handling

### Graceful Degradation:
1. **No Astrometry Solution:** Falls back to filename patterns
2. **No Filename Match:** Shows detection-only image
3. **No Catalog Stars:** Shows detection with warning
4. **No Detections:** Returns error message

### Common Error Cases:
- Overexposed/underexposed images
- Out-of-focus stars (too large/diffuse)
- Non-astronomical images
- Network issues with astrometry.net
- Missing catalog file

## Configuration Parameters

### Star Detection:
- `threshold_value`: 50-200 (default 120)
- `min_area`: 1-10 pixels
- `max_area`: 100-2000 pixels  
- `gaussian_blur`: 1-7 (odd numbers)

### Matching:
- `distance_threshold`: 0.1-2.0 degrees (default 0.8)
- `search_radius_multiplier`: 1-4 (default 2)

### Performance:
- `MAX_CONTENT_LENGTH`: 16MB file upload limit
- Timeout handling for astrometry.net requests
- Database connection pooling for multiple requests

This algorithm provides a robust, automated solution for astronomical star identification, balancing accuracy with processing speed for real-time web applications.


waitress-serve --host=0.0.0.0 --port=5000 flask_star_identifier:app