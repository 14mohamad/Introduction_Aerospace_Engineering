Link Video : 
https://drive.google.com/file/d/1Yt_HVLWzAw-5fVwM5HqzdvAEGkCcygQj/view?usp=drive_link

# Star Identifier

An automated star identification system that can detect stars in astronomical images and identify them using the Yale Bright Star Catalog (YBC5). The system uses computer vision techniques for star detection and plate solving for coordinate determination.

## Features

- **Automated Star Detection**: Uses OpenCV to detect bright objects in astronomical images
- **Plate Solving**: Integrates with Astrometry.net API for automatic coordinate determination
- **Fallback Pattern Matching**: Attempts to identify sky regions from filename patterns
- **Star Catalog Integration**: Uses Yale Bright Star Catalog (YBC5) for star identification
- **Visual Annotations**: Creates annotated images showing identified stars with labels
- **Web Interface**: Easy-to-use web interface for uploading and processing images
- **API Endpoints**: Multiple endpoints for different use cases

## Requirements

### Python Dependencies

```bash
pip install flask opencv-python numpy pandas matplotlib sqlite3 astroquery werkzeug
```

### Optional Dependencies

- `astroquery` - For Astrometry.net integration (automatic plate solving)
- `waitress` - For production deployment

### System Requirements

- Python 3.7+
- Minimum 1GB RAM (for image processing)
- Internet connection (for Astrometry.net API)

## Installation

1. **Clone or download the project files**:
   - `star_algorithm.py` - Core star identification algorithm
   - `flask_star_identifier.py` - Flask web server
   - `README.md` - This file

2. **Install dependencies**:
   ```bash
   pip install flask opencv-python numpy pandas matplotlib astroquery werkzeug waitress
   ```

3. **Set up the star catalog**:
   - Create a `data/` directory in your project folder
   - Download the Yale Bright Star Catalog (YBC5) file
   - Place it at `data/YBC5`
   - The system will automatically create a SQLite database on first run

## Usage

### Production Deployment (Recommended)

```bash
waitress-serve --host=0.0.0.0 --port=5000 flask_star_identifier:app
```

This will start the server at `http://localhost:5000`

### Development Mode

```bash
python flask_star_identifier.py
```

### Web Interface

1. Open your browser to `http://localhost:5000`
2. Upload an astronomical image (PNG, JPG, JPEG, GIF, BMP)
3. Adjust the threshold value if needed (50-200, default: 120)
4. Click "Identify Stars"
5. View the annotated result with identified stars

### API Endpoints

#### `/identify` (POST)
Returns JSON with base64-encoded annotated image and identification results.

**Parameters:**
- `image`: Image file
- `threshold`: Detection threshold (optional, default: 120)

**Response:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "info": {
    "stars_detected": 15,
    "stars_identified": 8,
    "identification_rate": "53.3%",
    "solve_method": "astrometry.net",
    "image_center": {"ra": 85.0, "dec": -1.0},
    "field_of_view": 15.0,
    "identified_stars": [...]
  }
}
```

#### `/identify_fast` (POST)
Returns JSON with image URL and identification results (faster for large images).

#### `/identify_raw` (POST)
Returns the annotated image directly as a PNG file.

### Example cURL Usage

```bash
curl -X POST -F "image=@your_star_image.jpg" -F "threshold=120" http://localhost:5000/identify
```

## How It Works

### 1. Star Detection
- Converts image to grayscale
- Applies Gaussian blur to reduce noise
- Uses binary thresholding to identify bright objects
- Filters objects by area to eliminate noise and cosmic rays

### 2. Coordinate Determination
The system tries multiple methods to determine sky coordinates:

1. **Astrometry.net API** (Primary method)
   - Uploads image to Astrometry.net for plate solving
   - Returns precise RA/Dec coordinates and field of view

2. **Filename Pattern Matching** (Fallback)
   - Recognizes common astronomical object names in filenames
   - Supported patterns: orion, m42, m31, andromeda, pleiades, m45, big_dipper, ursa_major, polaris, southern_cross, crux

### 3. Star Catalog Matching
- Searches Yale Bright Star Catalog for stars in the image region
- Converts pixel coordinates to sky coordinates
- Matches detected stars with catalog entries using position tolerance

### 4. Visual Annotation
- Creates annotated image showing all detected stars
- Highlights identified stars with green circles
- Adds labels with star names, HR numbers, and magnitudes

## Configuration

### Detection Parameters

You can adjust star detection by modifying these parameters in `star_algorithm.py`:

- `gaussian_blur`: Blur kernel size (default: 3)
- `threshold_value`: Binary threshold (default: 120)
- `min_area`: Minimum star area in pixels (default: 1)
- `max_area`: Maximum star area in pixels (default: 800)

### Astrometry.net Setup

The system includes a default API key for Astrometry.net. For production use, you should:

1. Register at [nova.astrometry.net](https://nova.astrometry.net)
2. Get your API key
3. Replace the API key in `star_algorithm.py`:
   ```python
   ast.api_key = 'your_api_key_here'
   ```

## Supported Image Formats

- PNG
- JPEG/JPG
- GIF
- BMP
- Maximum file size: 16MB

## File Structure

```
star-identifier/
├── star_algorithm.py          # Core identification algorithm
├── flask_star_identifier.py   # Web server
├── README.md                  # Documentation
├── data/
│   └── YBC5                   # Yale Bright Star Catalog
├── uploads/                   # Temporary image storage
└── complete_stars.db          # SQLite catalog database
```

## Troubleshooting

### Common Issues

1. **"No stars detected"**
   - Try lowering the threshold value
   - Ensure the image has sufficient contrast
   - Check that stars are bright enough in the image

2. **"Could not determine sky coordinates"**
   - Ensure internet connection for Astrometry.net
   - Try renaming the file to include astronomical object names
   - The system will still show detected stars without identification

3. **"No catalog stars found in this region"**
   - The image may be pointing to a region with few bright stars
   - Check that the YBC5 catalog file is properly installed

4. **Database errors**
   - Delete `complete_stars.db` to force recreation
   - Ensure the `data/YBC5` file is accessible

### Performance Tips

- Use images with resolution between 1000-4000 pixels for best results
- JPEGs are processed faster than PNGs
- Lower threshold values detect more stars but may include noise

## Limitations

- Only identifies stars in the Yale Bright Star Catalog (~9,000 brightest stars)
- Requires either internet connection or recognizable filename patterns
- Works best with wide-field astronomical images
- Performance depends on image quality and star density

## License

This project is provided as-is for educational and research purposes. Please ensure you have proper rights to use any star catalog data.
