import numpy as np
from collections import defaultdict
import json

def process_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox

    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min

    return x, y, w, h

def compute_stats(dataset):
    data_by_type = defaultdict(list)

    for sample in dataset:
        canvas = sample.get("canvas_size")
        if not canvas or len(canvas) < 2:
            continue
        canvas_height = float(canvas[1])
        if canvas_height <= 0:
            continue

        for elem in sample.get("elements", []):
            bbox_features = process_bbox(elem["bbox"])
            font_size = float(elem.get("font_size", 0.0) or 0.0)
            font_size_norm = font_size / canvas_height
            features = list(bbox_features) + [font_size_norm]
            data_by_type[elem["type"]].append(features)

    stats = {}

    for elem_type, features in data_by_type.items():
        X = np.array(features)
        X = np.clip(X, 0, 1)
        
        n_samples = X.shape[0]
        mu = np.mean(X, axis=0)
        
        if n_samples < 2:
            # Fallback to isotropic prior for single samples (same as DEFAULT_STATS)
            print(f"Warning: {elem_type} has only {n_samples} sample(s), using default isotropic covariance")
            cov = (0.1**2) * np.eye(5)
        else:
            cov = np.cov(X.T)
            
        # Regularization to ensure positive definite and invertible
        cov += 1e-6 * np.eye(cov.shape[0])
        
        try:
            cov_inv = np.linalg.inv(cov)
            cov_det = np.linalg.det(cov)
        except np.linalg.LinAlgError:
            # Fallback if inversion still fails (shouldn't happen with regularization, but safety first)
            print(f"Warning: {elem_type} covariance matrix singular, using pseudo-inverse")
            cov_inv = np.linalg.pinv(cov)
            cov_det = np.linalg.det(cov + 1e-4 * np.eye(5))  # Approximate with stronger regularization
            
        stats[elem_type] = {
            "mu": mu,
            "cov": cov,
            "cov_inv": cov_inv,
            "cov_det": cov_det,
        }

    return stats

def save_stats(stats, path):
    np.save(path, stats, allow_pickle=True)


def load_stats(path):
    return np.load(path, allow_pickle=True).item()


if __name__ == "__main__":
    with open("genposter_5000_images.json", "r") as f:
        dataset = json.load(f)

    stats = compute_stats(dataset)
    print("All the element types: ", stats.keys())
    out_path = "genposter_5000_images_stats.npy"
    save_stats(stats, out_path)
    print(f"Saved stats for {len(stats)} types to {out_path}")
    print("Load: stats = load_stats(%r)" % out_path)