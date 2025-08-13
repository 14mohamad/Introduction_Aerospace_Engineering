"""
Standalone complete star identifier with integrated astrometry and catalog
No hardcoded coordinates - uses proper fallback mechanisms
"""
import cv2
import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import os
import time
from typing import Optional, Tuple, List, Dict
import json

try:
    from astroquery.astrometry_net import AstrometryNet
    ASTROMETRY_AVAILABLE = True
except ImportError:
    ASTROMETRY_AVAILABLE = False

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

class StarIdentificationError(Exception):
    """Custom exception for star identification errors"""
    pass

def detect_stars(image_path, output_csv_path, 
                 gaussian_blur=3,  
                 threshold_value=70,
                 min_area=1, 
                 max_area=800):
    """
    Detect stars in an image and save (x, y, radius, brightness) to a CSV file.
    """
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Cannot load image at {image_path}")

    # Preprocess the image
    blurred = cv2.GaussianBlur(img, (gaussian_blur, gaussian_blur), 0)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours (bright regions)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stars_data = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Compute center and radius
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            brightness = img[int(y), int(x)]  # brightness from original (not blurred) image
            stars_data.append((x, y, radius, brightness))

    if not stars_data:
        print(f"No stars detected in {image_path}.")
    else:
        # Save to CSV
        folder = os.path.dirname(output_csv_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        df = pd.DataFrame(stars_data, columns=["x", "y", "r", "b"])
        df.to_csv(output_csv_path, index=False)
        print(f"Detected {len(df)} stars in {image_path}. Saved to {output_csv_path}")

    return stars_data

class CompleteCatalog:
    def __init__(self, ybc5_path='data/YBC5'):
        self.conn = sqlite3.connect('complete_stars.db')
        self.ybc5_path = ybc5_path
        self.create_and_populate()
    
    def parse_ybc5_line(self, line):
        """Parse a single line from YBC5 catalog using BSC5 format"""
        if len(line) < 147:
            return None
            
        try:
            # HR number (bytes 0-4)
            hr_str = line[0:4].strip()
            hr = int(hr_str) if hr_str else 0
            
            # Name (bytes 4-14)
            name = line[4:14].strip() or f"HR{hr}"
            
            # RA coordinates (bytes 75-83)
            ra_h = int(line[75:77]) if line[75:77].strip() else 0
            ra_m = int(line[77:79]) if line[77:79].strip() else 0
            ra_s = float(line[79:83]) if line[79:83].strip() else 0
            ra_deg = 15.0 * (ra_h + ra_m / 60.0 + ra_s / 3600.0)
            
            # Dec coordinates (bytes 83-90)
            sign = -1 if line[83] == "-" else 1
            dec_d = int(line[84:86]) if line[84:86].strip() else 0
            dec_m = int(line[86:88]) if line[86:88].strip() else 0
            dec_s = int(line[88:90]) if line[88:90].strip() else 0
            dec_deg = sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
            
            # Visual magnitude (bytes 102-107)
            vmag_raw = line[102:107].strip()
            if vmag_raw and vmag_raw != "99.99":
                try:
                    vmag = float(vmag_raw)
                except:
                    vmag = 99.0
            else:
                vmag = 99.0
            
            # Spectral type (bytes 127-147)
            spectral_type = line[127:147].strip() if len(line) >= 147 else ""
            
            return {
                'hr': hr,
                'name': name,
                'ra': ra_deg,
                'dec': dec_deg,
                'magnitude': vmag,
                'spectral_type': spectral_type
            }
            
        except Exception:
            return None
    
    def load_ybc5_catalog(self):
        """Load stars from YBC5 catalog file"""
        if not os.path.exists(self.ybc5_path):
            print(f"Warning: YBC5 file not found at {self.ybc5_path}")
            return []
        
        stars = []
        print("Loading Yale Bright Star Catalog...")
        
        with open(self.ybc5_path, 'r', encoding='latin-1') as f:
            for line_num, line in enumerate(f):
                star = self.parse_ybc5_line(line)
                if star:
                    stars.append(star)
                
                if line_num % 1000 == 0:
                    print(f"Processed {line_num} lines, found {len(stars)} valid stars")
        
        print(f"Loaded {len(stars)} stars from YBC5 catalog")
        return stars
    
    def create_and_populate(self):
        cursor = self.conn.cursor()
        
        # Drop existing table to ensure correct schema
        cursor.execute('DROP TABLE IF EXISTS stars')
        
        # Create table with YBC5 fields
        cursor.execute('''
            CREATE TABLE stars (
                hr INTEGER PRIMARY KEY,
                name TEXT,
                ra REAL,
                dec REAL,
                magnitude REAL,
                spectral_type TEXT
            )
        ''')
        
        print("Created new star catalog table")
        
        # Load from YBC5 file
        stars = self.load_ybc5_catalog()
        
        if not stars:
            print("ERROR: Failed to load YBC5 catalog - no star identification possible")
            return
        
        # Insert all stars
        for star in stars:
            cursor.execute('''
                INSERT OR REPLACE INTO stars (hr, name, ra, dec, magnitude, spectral_type) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (star['hr'], star['name'], star['ra'], star['dec'], 
                  star['magnitude'], star['spectral_type']))
        
        self.conn.commit()
        print(f"Loaded {len(stars)} stars into database")
    
    def find_stars_in_region(self, ra, dec, radius):
        """Find all stars within radius degrees of given position"""
        cursor = self.conn.cursor()
        
        # Simple box search
        cursor.execute('''
            SELECT * FROM stars 
            WHERE ra BETWEEN ? AND ? 
            AND dec BETWEEN ? AND ?
            ORDER BY magnitude
        ''', (ra - radius, ra + radius, dec - radius, dec + radius))
        
        results = cursor.fetchall()
        return pd.DataFrame(results, columns=['hr', 'name', 'ra', 'dec', 'magnitude', 'spectral_type'])

def check_fits_coordinates(image_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Check if image is a FITS file with WCS coordinates
    Returns: (ra, dec, radius) or (None, None, None)
    """
    if not ASTROPY_AVAILABLE:
        return None, None, None
        
    if not image_path.lower().endswith(('.fits', '.fit')):
        return None, None, None
        
    try:
        with fits.open(image_path) as hdul:
            header = hdul[0].header
            wcs = WCS(header)
            
            # Get image center
            ny, nx = hdul[0].data.shape[:2]
            center_x, center_y = nx / 2, ny / 2
            
            # Convert pixel to world coordinates
            ra, dec = wcs.pixel_to_world_values(center_x, center_y)
            
            # Estimate field of view
            corner_ra, corner_dec = wcs.pixel_to_world_values(0, 0)
            radius = np.sqrt((ra - corner_ra)**2 + (dec - corner_dec)**2)
            
            print(f"Found WCS coordinates in FITS header: RA={ra:.3f}°, Dec={dec:.3f}°")
            return float(ra), float(dec), float(radius)
            
    except Exception as e:
        print(f"Could not extract WCS from FITS: {e}")
        return None, None, None

def solve_image_astrometry(image_path: str, timeout: int = 180) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Solve an image using Astrometry.net API
    Returns: (ra, dec, radius) or (None, None, None) if failed
    """
    if not ASTROMETRY_AVAILABLE:
        print("Astrometry.net not available. Install with: pip install astroquery")
        return None, None, None
        
    ast = AstrometryNet()
    
    # Check for API key in environment variable or config file
    api_key = os.environ.get('ASTROMETRY_API_KEY')
    if not api_key and os.path.exists('astrometry_config.json'):
        with open('astrometry_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_key')
    
    if not api_key:
        print("No Astrometry.net API key found!")
        print("Please either:")
        print("1. Set environment variable: export ASTROMETRY_API_KEY='your_key'")
        print("2. Create astrometry_config.json with: {\"api_key\": \"your_key\"}")
        print("3. Get a free API key from: http://nova.astrometry.net/api_help")
        return None, None, None
    
    ast.api_key = api_key
    
    try:
        print(f"Solving {image_path} with Astrometry.net (timeout: {timeout}s)...")
        
        # Set solving parameters for better results
        kwargs = {
            'solve_timeout': timeout,
            'publicly_visible': 'n',
            'scale_units': 'degwidth',
            'scale_lower': 0.1,  # minimum field width in degrees
            'scale_upper': 180.0,  # maximum field width in degrees
            'scale_est': 2.0,  # estimated field width
            'scale_err': 50.0,  # error percentage
        }
        
        wcs_header = ast.solve_from_image(image_path, **kwargs)
        
        if wcs_header:
            ra = wcs_header.get('CRVAL1', wcs_header.get('RA'))
            dec = wcs_header.get('CRVAL2', wcs_header.get('DEC'))
            radius = wcs_header.get('RADIUS', 1.0)
            
            print(f"Success! RA: {ra:.3f}°, Dec: {dec:.3f}°, Radius: {radius:.3f}°")
            return float(ra), float(dec), float(radius)
        else:
            print("Solve failed - no WCS header returned")
            return None, None, None
        
    except Exception as e:
        print(f"Astrometry.net error: {e}")
        return None, None, None

def get_user_coordinates() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Interactively ask user for approximate coordinates
    Returns: (ra, dec, radius) or (None, None, None) if user cancels
    """
    print("\n=== Manual Coordinate Entry ===")
    print("Please provide approximate sky coordinates for this image.")
    print("(You can find these from your observation log, planetarium software, etc.)")
    
    try:
        print("\nRight Ascension (RA):")
        print("  Format: degrees (0-360) or HH:MM:SS")
        ra_input = input("  RA: ").strip()
        
        if ':' in ra_input:
            # Parse HH:MM:SS format
            parts = ra_input.split(':')
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2] if len(parts) > 2 else 0)
            ra = 15.0 * (h + m/60.0 + s/3600.0)
        else:
            ra = float(ra_input)
        
        print("\nDeclination (Dec):")
        print("  Format: degrees (-90 to +90) or ±DD:MM:SS")
        dec_input = input("  Dec: ").strip()
        
        if ':' in dec_input:
            # Parse ±DD:MM:SS format
            sign = -1 if dec_input.startswith('-') else 1
            dec_input = dec_input.lstrip('+-')
            parts = dec_input.split(':')
            d, m, s = float(parts[0]), float(parts[1]), float(parts[2] if len(parts) > 2 else 0)
            dec = sign * (d + m/60.0 + s/3600.0)
        else:
            dec = float(dec_input)
        
        print("\nField of view radius (degrees):")
        print("  (Rough estimate - e.g., 1 for narrow field, 5-10 for wide field)")
        radius = float(input("  Radius [default=2.0]: ").strip() or "2.0")
        
        # Validate ranges
        if not (0 <= ra <= 360):
            raise ValueError("RA must be between 0 and 360 degrees")
        if not (-90 <= dec <= 90):
            raise ValueError("Dec must be between -90 and +90 degrees")
        if not (0.1 <= radius <= 90):
            raise ValueError("Radius must be between 0.1 and 90 degrees")
        
        print(f"\nUsing coordinates: RA={ra:.3f}°, Dec={dec:.3f}°, Radius={radius:.3f}°")
        return ra, dec, radius
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nInvalid input or cancelled: {e}")
        return None, None, None

def save_solution_cache(image_path: str, ra: float, dec: float, radius: float):
    """Save solved coordinates to a cache file"""
    cache_file = image_path + '.coords'
    with open(cache_file, 'w') as f:
        json.dump({'ra': ra, 'dec': dec, 'radius': radius}, f)
    print(f"Saved coordinates to {cache_file}")

def load_solution_cache(image_path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Load previously solved coordinates from cache"""
    cache_file = image_path + '.coords'
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded cached coordinates from {cache_file}")
            return data['ra'], data['dec'], data['radius']
        except:
            pass
    return None, None, None

def identify_stars_complete(image_path: str, 
                          force_manual: bool = False,
                          use_cache: bool = True,
                          solver_timeout: int = 180) -> List[Dict]:
    """
    Complete star identification pipeline
    
    Parameters:
    - image_path: Path to the image file
    - force_manual: Skip automatic solving and go straight to manual input
    - use_cache: Use cached coordinates if available
    - solver_timeout: Timeout for astrometry.net solver in seconds
    
    Returns:
    - List of matched stars with their information
    """
    print(f"\n=== Identifying stars in {image_path} ===\n")
    
    # 1. Detect stars
    csv_path = image_path.replace('.jpg', '_detected.csv').replace('.png', '_detected.csv')
    print("1. Detecting stars in image...")
    stars = detect_stars(image_path, csv_path, threshold_value=120)
    print(f"   Found {len(stars)} stars")
    
    if not stars:
        print("\nNo stars detected in image. Try adjusting detection parameters.")
        return []
    
    # 2. Get sky coordinates
    print("\n2. Determining sky coordinates...")
    ra, dec, radius = None, None, None
    
    # Try multiple methods in order
    coordinate_methods = []
    
    if not force_manual:
        if use_cache:
            coordinate_methods.append(("cached solution", lambda: load_solution_cache(image_path)))
        coordinate_methods.append(("FITS header", lambda: check_fits_coordinates(image_path)))
        coordinate_methods.append(("Astrometry.net", lambda: solve_image_astrometry(image_path, solver_timeout)))
    
    coordinate_methods.append(("manual input", get_user_coordinates))
    
    for method_name, method_func in coordinate_methods:
        print(f"\n   Trying {method_name}...")
        ra, dec, radius = method_func()
        if ra is not None:
            print(f"   Success with {method_name}!")
            break
    
    if ra is None:
        raise StarIdentificationError(
            "Could not determine sky coordinates for this image.\n"
            "Please ensure you have:\n"
            "1. A valid Astrometry.net API key, or\n"
            "2. FITS files with WCS headers, or\n"
            "3. Knowledge of approximate coordinates"
        )
    
    # Cache the solution for future use
    if use_cache and ra is not None:
        save_solution_cache(image_path, ra, dec, radius)
    
    print(f"\n   Image center: RA={ra:.3f}°, Dec={dec:.3f}°")
    print(f"   Field of view: {radius*2:.2f}°")
    
    # 3. Load catalog and find nearby stars
    print("\n3. Loading star catalog...")
    catalog = CompleteCatalog()
    catalog_stars = catalog.find_stars_in_region(ra, dec, radius * 2)
    print(f"   Found {len(catalog_stars)} catalog stars in region")
    
    if len(catalog_stars) == 0:
        print("\nNo catalog stars found in this region!")
        print("This might indicate:")
        print("- Incorrect coordinates")
        print("- Very faint star field")
        print("- Problem with the star catalog")
        catalog.conn.close()
        return []
    
    # 4. Match detected stars with catalog
    print("\n4. Matching detected stars with catalog...")
    detected_df = pd.read_csv(csv_path)
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Calculate pixel scale
    pixel_scale_deg_per_pixel = (radius * 2) / min(width, height)
    
    matches = []
    used_catalog_stars = set()
    print(f"   Pixel scale: {pixel_scale_deg_per_pixel*3600:.2f} arcsec/pixel")
    
    # Create list of detected stars with their sky coordinates
    detected_with_coords = []
    for idx, star in detected_df.iterrows():
        # Convert pixel to sky coordinates
        x_pixels_from_center = star['x'] - width/2
        y_pixels_from_center = -(star['y'] - height/2)
        
        # Convert to degrees offset from center
        x_deg_offset = x_pixels_from_center * pixel_scale_deg_per_pixel
        y_deg_offset = y_pixels_from_center * pixel_scale_deg_per_pixel
        
        # Apply proper spherical coordinate transformation
        star_ra = ra + x_deg_offset / np.cos(np.radians(dec))
        star_dec = dec + y_deg_offset
        
        detected_with_coords.append({
            'idx': idx,
            'x': star['x'],
            'y': star['y'],
            'ra': star_ra,
            'dec': star_dec,
            'brightness': star['b']
        })
    
    # Match stars (prioritize brighter catalog stars)
    matching_threshold = 0.5  # degrees
    
    for _, cat_star in catalog_stars.iterrows():
        if cat_star['hr'] in used_catalog_stars:
            continue
            
        best_detected = None
        min_dist = float('inf')
        
        # Find closest detected star to this catalog star
        for det_star in detected_with_coords:
            dist = np.sqrt((det_star['ra'] - cat_star['ra'])**2 + 
                          (det_star['dec'] - cat_star['dec'])**2)
            if dist < min_dist and dist < matching_threshold:
                min_dist = dist
                best_detected = det_star
        
        if best_detected is not None:
            matches.append({
                'x': best_detected['x'],
                'y': best_detected['y'],
                'name': cat_star['name'],
                'hr': cat_star['hr'],
                'magnitude': cat_star['magnitude'],
                'spectral_type': cat_star['spectral_type'],
                'distance': min_dist,
                'brightness': best_detected['brightness']
            })
            used_catalog_stars.add(cat_star['hr'])
            # Remove matched detected star
            detected_with_coords = [d for d in detected_with_coords if d['idx'] != best_detected['idx']]
    
    print(f"   Matched {len(matches)} stars")
    
    # 5. Visualize results
    print("\n5. Visualizing results...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img_rgb)
    
    # Plot all detected stars
    plt.scatter(detected_df['x'], detected_df['y'], s=50, c='red', marker='o', 
                edgecolors='white', linewidths=1, alpha=0.5, label='Unidentified')
    
    # Plot and label matched stars
    for match in matches:
        plt.scatter(match['x'], match['y'], s=200, facecolors='none', 
                   edgecolors='lime', linewidths=2)
        plt.annotate(f"{match['name']}\nHR{match['hr']} (mag {match['magnitude']:.1f})",
                    xy=(match['x'], match['y']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, color='yellow',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1))
    
    plt.title(f'Star Identification Results - {len(matches)} stars identified', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the annotated image
    output_path = image_path.replace('.jpg', '_identified.png').replace('.png', '_identified.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n   Saved annotated image to: {output_path}")
    
    plt.show()
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total stars detected: {len(detected_df)}")
    print(f"Stars identified: {len(matches)}")
    if len(detected_df) > 0:
        print(f"Identification rate: {len(matches)/len(detected_df)*100:.1f}%")
    
    if matches:
        print("\nIdentified stars (sorted by magnitude):")
        for match in sorted(matches, key=lambda x: x['magnitude']):
            print(f"\n  {match['name']} (HR{match['hr']})")
            print(f"    Magnitude: {match['magnitude']:.2f}")
            print(f"    Spectral type: {match['spectral_type']}")
            print(f"    Pixel position: ({match['x']:.1f}, {match['y']:.1f})")
            print(f"    Match distance: {match['distance']*3600:.1f} arcsec")
    
    # Save identification results
    results_file = image_path.replace('.jpg', '_results.json').replace('.png', '_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'image': image_path,
            'coordinates': {'ra': ra, 'dec': dec, 'radius': radius},
            'detected_stars': len(detected_df),
            'identified_stars': len(matches),
            'matches': matches
        }, f, indent=2)
    print(f"\nSaved detailed results to: {results_file}")
    
    catalog.conn.close()
    return matches

if __name__ == "__main__":
    import sys
    
    # Example usage with command line arguments
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = 'imgs/fr1.jpg'
    
    # Run with various options
    try:
        # Normal run - tries all automatic methods first
        matches = identify_stars_complete(image_path)
        
        # Force manual coordinate entry (skip automatic solving)
        # matches = identify_stars_complete(image_path, force_manual=True)
        
        # Don't use cached coordinates
        # matches = identify_stars_complete(image_path, use_cache=False)
        
    except StarIdentificationError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled by user")
        sys.exit(0)