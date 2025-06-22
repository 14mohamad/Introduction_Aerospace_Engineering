import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

def detect_stars(image_path, output_csv_path, 
                 gaussian_blur=3, 
                 threshold_value=200, 
                 min_area=3, 
                 max_area=500):
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

def plot_stars_side_by_side(image_path, csv_path, point_color=(1, 0, 0), radius=5):
    """
    Plots the original image and the detected stars image side by side.
    
    Args:
        image_path (str): Path to the original image.
        csv_path (str): Path to the CSV file with columns x, y.
        point_color (tuple): Color of the points (R, G, B) normalized to [0,1].
        radius (int): Radius of the plotted points.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Cannot load image at {image_path}")

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load detected stars
    stars_df = pd.read_csv(csv_path)

    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Show original image
    axs[0].imshow(img_rgb)
    axs[0].set_title(f"Original Image: {os.path.basename(image_path)}")
    axs[0].axis('off')

    # Show image with stars
    axs[1].imshow(img_rgb)
    axs[1].scatter(stars_df['x'], stars_df['y'], s=radius**2, c=[point_color], edgecolors='white')
    axs[1].set_title(f"Detected Stars: {os.path.basename(image_path)}")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def load_stars(csv_path):
    df = pd.read_csv(csv_path)
    return df[['x', 'y']].values

def compute_triangle_distances(points):
    d1 = np.linalg.norm(points[0] - points[1])
    d2 = np.linalg.norm(points[0] - points[2])
    d3 = np.linalg.norm(points[1] - points[2])
    return sorted([d1, d2, d3])

def compute_quadrilateral_distances(points):
    return sorted([
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[0] - points[2]),
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]),
        np.linalg.norm(points[1] - points[3]),
        np.linalg.norm(points[2] - points[3]),
    ])

def normalize_distances(dists):
    max_val = max(dists)
    return [d / max_val if max_val != 0 else 0 for d in dists]

def build_triangle_features_knn(stars, k=5):
    triangles = []
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(stars)
    _, indices = nbrs.kneighbors(stars)
    for center, neighbors in enumerate(indices):
        for pair in combinations(neighbors[1:], 2):
            inds = (center, pair[0], pair[1])
            points = stars[list(inds)]
            dists = compute_triangle_distances(points)
            norm_dists = normalize_distances(dists)
            triangles.append((inds, norm_dists))
    return triangles

def build_quadrilateral_features_knn(stars, k=5):
    quads = []
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(stars)
    _, indices = nbrs.kneighbors(stars)
    for center, neighbors in enumerate(indices):
        for triplet in combinations(neighbors[1:], 3):
            inds = (center, triplet[0], triplet[1], triplet[2])
            points = stars[list(inds)]
            dists = compute_quadrilateral_distances(points)
            norm_dists = normalize_distances(dists)
            quads.append((inds, norm_dists))
    return quads

def match_features_best(features_small, features_large, tolerance):
    matches = []
    for inds_small, norm_dists_small in features_small:
        best_error = float('inf')
        best_match = None
        for inds_large, norm_dists_large in features_large:
            error = np.mean([abs(s - l) for s, l in zip(norm_dists_small, norm_dists_large)])
            if error < tolerance and error < best_error:
                best_error = error
                best_match = inds_large
        if best_match is not None:
            matches.append((inds_small, best_match))
    return matches

def find_star_matches(small_csv, large_csv, use_quads=True, k=4, tolerance_triangle=0.03, tolerance_quad=0.02):
    stars_small = load_stars(small_csv)
    stars_large = load_stars(large_csv)
    if use_quads:
        feats_small = build_quadrilateral_features_knn(stars_small, k)
        feats_large = build_quadrilateral_features_knn(stars_large, k)
        tol = tolerance_quad
    else:
        feats_small = build_triangle_features_knn(stars_small, k)
        feats_large = build_triangle_features_knn(stars_large, k)
        tol = tolerance_triangle
    matches = match_features_best(feats_small, feats_large, tol)
    votes = defaultdict(lambda: defaultdict(int))
    for s_inds, l_inds in matches:
        for s, l in zip(s_inds, l_inds):
            votes[s][l] += 1
    final_matches = {}
    used_large = set()
    for s in votes:
        sorted_votes = sorted(votes[s].items(), key=lambda x: -x[1])
        for l, _ in sorted_votes:
            if l not in used_large:
                final_matches[s] = l
                used_large.add(l)
                break
    return [(s, l) for s, l in final_matches.items()]

def visualize_matches_inline(small_img_path, small_csv, large_img_path, large_csv, matches):
    img1 = cv2.imread(small_img_path)
    img2 = cv2.imread(large_img_path)
    df1 = pd.read_csv(small_csv)
    df2 = pd.read_csv(large_csv)
    for s, l in matches:
        x1, y1 = df1.loc[s, ['x', 'y']]
        x2, y2 = df2.loc[l, ['x', 'y']]
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img1, (int(x1), int(y1)), 12, color, -1)
        cv2.circle(img2, (int(x2), int(y2)), 12, color, -1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    axs[0].imshow(img1)
    axs[0].set_title("Image1")
    axs[1].imshow(img2)
    axs[1].set_title("Image2")
    for ax in axs: ax.axis("off")
    plt.show()

if __name__ == "__main__":
    # Detect stars for all 4 provided images
    image_files = [
        ("imgs/fr1.jpg", "imgs/stars_fr1.csv"),
        ("imgs/fr2.jpg", "imgs/stars_fr2.csv"),
        ("imgs/ST_db1.png", "imgs/stars_ST_db1.csv"),
        ("imgs/ST_db2.png", "imgs/stars_ST_db2.csv")
    ]

    for img_path, csv_path in image_files:
        detect_stars(img_path, csv_path)

    # Check detected stars
    image_csv_pairs = [
        ("fr1.jpg", "stars_fr1.csv"),
        ("fr2.jpg", "stars_fr2.csv"),
        ("ST_db1.png", "stars_ST_db1.csv"),
        ("ST_db2.png", "stars_ST_db2.csv"),
    ]

    for img_path, csv_path in image_csv_pairs:
        plot_stars_side_by_side(img_path, csv_path)

    # Images and corresponding CSVs
    images = {
        "fr1": ("fr1.jpg", "stars_fr1.csv"),
        "fr2": ("fr2.jpg", "stars_fr2.csv"),
        "ST_db1": ("ST_db1.png", "stars_ST_db1.csv"),
        "ST_db2": ("ST_db2.png", "stars_ST_db2.csv")
    }

    results = []

    # Matching across all ordered pairs
    for small_name in images:
        for large_name in images:
            if small_name == large_name:
                continue

            print(f"\nMatching {small_name} → {large_name}")

            small_img, small_csv = images[small_name]
            large_img, large_csv = images[large_name]

            matches = find_star_matches(small_csv, large_csv)

            # Visualize the matches
            visualize_matches_inline(small_img, small_csv, large_img, large_csv, matches)