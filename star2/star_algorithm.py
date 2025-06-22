"""
Core star identification algorithm
Handles star detection, catalog matching, and coordinate solving
"""
import cv2
import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import tempfile

try:
    from astroquery.astrometry_net import AstrometryNet
    ASTROMETRY_AVAILABLE = True
except ImportError:
    ASTROMETRY_AVAILABLE = False

def detect_stars(img, gaussian_blur=3, threshold_value=70, min_area=1, max_area=800):
    # gaussian_blur: to remove noise
    """Detect stars in an image and return star positions"""
    print(f"🔍 STEP 1: Starting star detection with threshold={threshold_value}...")
    
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
        
    blurred = cv2.GaussianBlur(img_gray, (gaussian_blur, gaussian_blur), 0)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stars_data = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            brightness = img_gray[int(y), int(x)]
            stars_data.append((x, y, radius, brightness))
    
    print(f"✅ STEP 1 Complete: Detected {len(stars_data)} potential stars from {len(contours)} bright spots")
    return stars_data

class CompleteCatalog:
    def __init__(self, ybc5_path='data/YBC5'):
        self.conn = sqlite3.connect('complete_stars.db', check_same_thread=False)
        self.ybc5_path = ybc5_path
        if not self._table_exists():
            self.create_and_populate()
    
    def _table_exists(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stars'")
        return cursor.fetchone() is not None
    
    #ybc5 to sqlite3
    def parse_ybc5_line(self, line):
        """Parse a single line from YBC5 catalog"""
        if len(line) < 147:
            return None
            
        try:
            hr_str = line[0:4].strip()
            hr = int(hr_str) if hr_str else 0
            
            name = line[4:14].strip() or f"HR{hr}"
            
            ra_h = int(line[75:77]) if line[75:77].strip() else 0
            ra_m = int(line[77:79]) if line[77:79].strip() else 0
            ra_s = float(line[79:83]) if line[79:83].strip() else 0
            ra_deg = 15.0 * (ra_h + ra_m / 60.0 + ra_s / 3600.0)
            
            sign = -1 if line[83] == "-" else 1
            dec_d = int(line[84:86]) if line[84:86].strip() else 0
            dec_m = int(line[86:88]) if line[86:88].strip() else 0
            dec_s = int(line[88:90]) if line[88:90].strip() else 0
            dec_deg = sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
            
            vmag_raw = line[102:107].strip()
            if vmag_raw and vmag_raw != "99.99":
                try:
                    vmag = float(vmag_raw)
                except:
                    vmag = 99.0
            else:
                vmag = 99.0
            
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
        with open(self.ybc5_path, 'r', encoding='latin-1') as f:
            for line in f:
                star = self.parse_ybc5_line(line)
                if star:
                    stars.append(star)
        
        return stars
    
    def create_and_populate(self):
        cursor = self.conn.cursor()
        
        cursor.execute('DROP TABLE IF EXISTS stars')
        
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
        
        stars = self.load_ybc5_catalog()
        
        for star in stars:
            cursor.execute('''
                INSERT OR REPLACE INTO stars (hr, name, ra, dec, magnitude, spectral_type) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (star['hr'], star['name'], star['ra'], star['dec'], 
                  star['magnitude'], star['spectral_type']))
        
        self.conn.commit()
    
    def find_stars_in_region(self, ra, dec, radius):
        """Find all stars within radius degrees of given position"""
        print(f"📚 STEP 3: Searching star catalog around RA={ra:.2f}°, Dec={dec:.2f}°, radius={radius:.2f}°...")
        
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT * FROM stars 
            WHERE ra BETWEEN ? AND ? 
            AND dec BETWEEN ? AND ?
            ORDER BY magnitude
        ''', (ra - radius, ra + radius, dec - radius, dec + radius))
        
        results = cursor.fetchall()
        stars_df = pd.DataFrame(results, columns=['hr', 'name', 'ra', 'dec', 'magnitude', 'spectral_type'])
        
        print(f"✅ STEP 3 Complete: Found {len(stars_df)} catalog stars in region (brightest: mag {stars_df['magnitude'].min():.1f})")
        return stars_df

# api call to Astrometry.net to get sky coordinates (RA, DEC)
def solve_image_astrometry(image_path):
    """Solve an image using Astrometry.net API"""
    print("🌍 STEP 2A: Attempting automatic plate solving via Astrometry.net...")
    
    if not ASTROMETRY_AVAILABLE:
        print("❌ Astrometry.net library not available")
        return None, None, None
        
    ast = AstrometryNet()
    ast.api_key = 'nuawxvldlgovsbwo'
    
    try:
        wcs_header = ast.solve_from_image(image_path)
        
        if wcs_header:
            ra = wcs_header.get('CRVAL1', wcs_header.get('RA'))
            dec = wcs_header.get('CRVAL2', wcs_header.get('DEC'))
            radius = wcs_header.get('RADIUS', 1.0)
            print(f"✅ STEP 2A Success: Found coordinates RA={ra:.2f}°, Dec={dec:.2f}°, FOV={radius:.2f}°")
            return ra, dec, radius
        else:
            print("❌ STEP 2A Failed: Astrometry.net could not solve image")
            return None, None, None
            
    except Exception as e:
        print(f"❌ STEP 2A Failed: Astrometry.net error - {str(e)}")
        return None, None, None

# fall back to guess coordinates from filename
def estimate_coordinates_from_filename(filename):
    """Try to guess sky region from filename patterns - ONLY as fallback"""
    print(f"🔍 STEP 2B: Attempting filename pattern matching for '{filename}'...")
    
    filename_lower = filename.lower()
    
    # Common astronomical regions with coordinates
    regions = {
        'orion': (85.0, -1.0, 15.0),
        'm42': (83.8, -5.4, 2.0),
        'm31': (10.7, 41.3, 3.0),
        'andromeda': (10.7, 41.3, 3.0),
        'pleiades': (56.9, 24.1, 2.0),
        'm45': (56.9, 24.1, 2.0),
        'big_dipper': (180.0, 55.0, 15.0),
        'ursa_major': (180.0, 55.0, 15.0),
        'polaris': (37.9, 89.3, 5.0),
        'southern_cross': (186.0, -60.0, 10.0),
        'crux': (186.0, -60.0, 10.0),
    }
    
    for name, (ra, dec, radius) in regions.items():
        if name in filename_lower:
            print(f"✅ STEP 2B Success: Matched '{name}' → RA={ra}°, Dec={dec}°, FOV={radius}°")
            return ra, dec, radius
    
    print("❌ STEP 2B Failed: No recognizable astronomical patterns in filename")
    return None, None, None

# Create an image showing only detected stars red circles
def create_detection_only_image(img, detected_df):
    """Create an annotated image showing only detected stars (no identification)"""
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Plot all detected stars as red circles
    plt.scatter(detected_df['x'], detected_df['y'], s=50, c='red', marker='o', 
                edgecolors='white', linewidths=1, alpha=0.7)
    
    plt.title(f'Star Detection Results - {len(detected_df)} stars detected (No identification available)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convert to image
    annotated_img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    return annotated_img

def identify_and_annotate_stars(img, filename="image.jpg", threshold_value=120):
    """Main function to identify stars and create annotated image"""
    # 1. Detect stars
    stars = detect_stars(img, threshold_value=threshold_value)
    
    if not stars:
        return img, {"error": "No stars detected", "stars_detected": 0}
    
    detected_df = pd.DataFrame(stars, columns=["x", "y", "r", "b"])
    
    # 2. Create temporary file for astrometry
    temp_path = None
    if ASTROMETRY_AVAILABLE:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            temp_path = tmp.name
    
    # 3. Get sky coordinates - try automatic first
    ra, dec, radius = solve_image_astrometry(temp_path) if temp_path else (None, None, None)
    solve_method = "astrometry.net"
    
    if temp_path and os.path.exists(temp_path):
        os.unlink(temp_path)
    
    # If automatic failed, try filename fallback
    if ra is None:
        ra, dec, radius = estimate_coordinates_from_filename(filename)
        if ra is not None:
            solve_method = "filename pattern"
    
    # If still no coordinates, return detection-only image
    if ra is None:
        annotated_img = create_detection_only_image(img, detected_df)
        return annotated_img, {
            "stars_detected": len(detected_df),
            "stars_identified": 0,
            "identification_rate": "0.0%",
            "solve_method": "none",
            "error": "Could not determine sky coordinates - showing detected stars only"
        }
    
    # 4. Load catalog and find nearby stars
    catalog = CompleteCatalog()
    catalog_stars = catalog.find_stars_in_region(ra, dec, radius * 2)
    
    # If no catalog stars in region, return detection-only image
    if len(catalog_stars) == 0:
        annotated_img = create_detection_only_image(img, detected_df)
        return annotated_img, {
            "stars_detected": len(detected_df),
            "stars_identified": 0,
            "identification_rate": "0.0%",
            "solve_method": solve_method,
            "image_center": {"ra": ra, "dec": dec},
            "field_of_view": radius * 2,
            "error": "No catalog stars found in this region - showing detected stars only"
        }
    
    # 5. Match detected stars with catalog
    print(f"🔄 STEP 4: Converting {len(detected_df)} pixel positions to sky coordinates...")
    
    height, width = img.shape[:2]
    pixel_scale_deg_per_pixel = (radius * 2) / min(width, height)
    
    matches = []
    used_catalog_stars = set()
    
    # Create list of detected stars with their sky coordinates
    detected_with_coords = []
    for idx, star in detected_df.iterrows():
        x_pixels_from_center = star['x'] - width/2
        y_pixels_from_center = -(star['y'] - height/2)
        
        x_deg_offset = x_pixels_from_center * pixel_scale_deg_per_pixel
        y_deg_offset = y_pixels_from_center * pixel_scale_deg_per_pixel
        
        star_ra = ra + x_deg_offset / np.cos(np.radians(dec))
        star_dec = dec + y_deg_offset
        
        detected_with_coords.append({
            'idx': idx,
            'x': star['x'],
            'y': star['y'],
            'ra': star_ra,
            'dec': star_dec
        })
    
    print(f"✅ STEP 4 Complete: Pixel scale = {pixel_scale_deg_per_pixel:.4f} deg/pixel, all positions transformed")
    
    print(f"🎯 STEP 5: Matching {len(detected_with_coords)} detected stars with {len(catalog_stars)} catalog stars...")
    
    # Find best matches
    for _, cat_star in catalog_stars.iterrows():
        if cat_star['hr'] in used_catalog_stars:
            continue
            
        best_detected = None
        min_dist = float('inf')
        
        for det_star in detected_with_coords:
            dist = np.sqrt((det_star['ra'] - cat_star['ra'])**2 + (det_star['dec'] - cat_star['dec'])**2)
            if dist < min_dist and dist < 0.8:
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
                'distance': min_dist
            })
            used_catalog_stars.add(cat_star['hr'])
            detected_with_coords = [d for d in detected_with_coords if d['idx'] != best_detected['idx']]
    
    identification_rate = (len(matches) / len(detected_df) * 100) if len(detected_df) > 0 else 0
    print(f"✅ STEP 5 Complete: Successfully matched {len(matches)} stars ({identification_rate:.1f}% identification rate)")
    
    # 6. Create annotated image
    print(f"🎨 STEP 6: Creating annotated star map with {len(matches)} identified stars...")
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
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
    
    title = f'Star Identification Results - {len(matches)} stars identified'
    if solve_method == "filename pattern":
        title += " (using filename pattern)"
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    
    # Convert to image
    annotated_img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    
    print(f"✅ STEP 6 Complete: Generated annotated image with visual results")
    
    # Create result info
    print(f"📊 FINAL RESULTS: {len(detected_df)} detected → {len(matches)} identified ({identification_rate:.1f}% success) via {solve_method}")
    
    result_info = {
        "stars_detected": len(detected_df),
        "stars_identified": len(matches),
        "identification_rate": f"{len(matches)/len(detected_df)*100:.1f}%",
        "solve_method": solve_method,
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
    
    return annotated_img, result_info

def create_annotated_image_cv2(img, detected_df, matches, solve_method=""):
    """Create annotated image using OpenCV (alternative to matplotlib)"""
    annotated_img = img.copy()
    
    # Draw detected stars (red circles)
    for _, star in detected_df.iterrows():
        x, y = int(star['x']), int(star['y'])
        cv2.circle(annotated_img, (x, y), 8, (0, 0, 255), 2)  # Red circle
        cv2.circle(annotated_img, (x, y), 3, (255, 255, 255), -1)  # White center
    
    # Draw identified stars (green circles with labels)
    for i, match in enumerate(matches):
        x, y = int(match['x']), int(match['y'])
        # Green circle for identified stars
        cv2.circle(annotated_img, (x, y), 12, (0, 255, 0), 3)
        
        # Create label text
        label = f"{match['name']}"
        magnitude_text = f"HR{match['hr']} (mag {match['magnitude']:.1f})"
        
        # Calculate text position (avoid going off screen)
        text_x = min(x + 15, img.shape[1] - 200)
        text_y = max(y - 10, 30)
        
        # Draw background rectangle for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (mag_w, mag_h), _ = cv2.getTextSize(magnitude_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        max_width = max(text_w, mag_w)
        
        # Black background with some transparency
        overlay = annotated_img.copy()
        cv2.rectangle(overlay, (text_x - 5, text_y - text_h - 5), 
                     (text_x + max_width + 10, text_y + mag_h + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_img, 0.3, 0, annotated_img)
        
        # Draw text labels
        cv2.putText(annotated_img, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Yellow text
        cv2.putText(annotated_img, magnitude_text, (text_x, text_y + text_h + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # White text
        
        # Draw arrow from circle to label
        cv2.arrowedLine(annotated_img, (x + 12, y), (text_x - 5, text_y - text_h//2), 
                       (0, 255, 255), 2, tipLength=0.3)
    
    # Add title text at top
    title_text = f'Star Identification Results - {len(matches)} stars identified'
    if solve_method == "filename pattern":
        title_text += " (filename pattern)"
    title_x = 20
    title_y = 40
    
    # Title background
    (title_w, title_h), _ = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    overlay = annotated_img.copy()
    cv2.rectangle(overlay, (title_x - 10, title_y - title_h - 10), 
                 (title_x + title_w + 20, title_y + 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, annotated_img, 0.2, 0, annotated_img)
    
    # Title text
    cv2.putText(annotated_img, title_text, (title_x, title_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    return annotated_img