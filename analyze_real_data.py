import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.linear_model import RANSACRegressor

def robust_linear_fit_current(x, y):
    """Current implementation with RANSAC-like init"""
    if len(x) < 2: return 0.0, 0.0
    n_pts = len(x)
    sub_size = min(n_pts, 500)
    indices = np.random.choice(n_pts, sub_size, replace=False)
    xs, ys = x[indices], y[indices]
    best_a, best_b = 0.0, 0.0
    max_inliers = -1
    for _ in range(50):
        pair_idx = np.random.choice(len(xs), 2, replace=False)
        x1, y1 = xs[pair_idx[0]], ys[pair_idx[0]]; x2, y2 = xs[pair_idx[1]], ys[pair_idx[1]]
        if x1 == x2: continue
        a = (y2 - y1) / (x2 - x1); b = y1 - a * x1
        y_range = np.ptp(ys); threshold = max(y_range * 0.05, 1.0)
        residuals = np.abs(a * xs + b - ys)
        inliers = np.sum(residuals < threshold)
        if inliers > max_inliers:
            max_inliers = inliers; best_a, best_b = a, b
    def model(params, x): return params[0] * x + params[1]
    def residuals_fn(params, x, y): return model(params, x) - y
    consensus_res = np.abs(best_a * xs + best_b - ys)
    f_scale = np.median(consensus_res) * 2.0
    if f_scale <= 0: f_scale = 1.0
    res = least_squares(residuals_fn, [best_a, best_b], args=(x, y), loss='soft_l1', f_scale=f_scale)
    return res.x

def solve_with_ransac_sklearn(x, y):
    ransac = RANSACRegressor()
    ransac.fit(x.reshape(-1, 1), y)
    return ransac.estimator_.coef_[0], ransac.estimator_.intercept_

def fit_percentile_clipping(x, y):
    """Clip 10-90 percentiles to remove furthest outliers before fitting"""
    x_low, x_high = np.percentile(x, [10, 90])
    y_low, y_high = np.percentile(y, [10, 90])
    mask = (x > x_low) & (x < x_high) & (y > y_low) & (y < y_high)
    x_clip, y_clip = x[mask], y[mask]
    if len(x_clip) < 2: return robust_linear_fit_current(x, y)
    return np.polyfit(x_clip, y_clip, 1)

# Load real data
df = pd.read_csv('scatter_491_s5_405_s5_frame45.csv')
bg = df[df['Label'] == 0]
x = bg['X'].values
y = bg['Y'].values

# Fits
a_ols, b_ols = np.polyfit(x, y, 1)
a_r_curr, b_r_curr = robust_linear_fit_current(x, y)
a_clip, b_clip = fit_percentile_clipping(x, y)

print(f"OLS: y = {a_ols:.3f}x + {b_ols:.1f}")
print(f"Current Robust: y = {a_r_curr:.3f}x + {b_r_curr:.1f}")
print(f"Percentile Clip: y = {a_clip:.3f}x + {b_clip:.1f}")

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(x, y, alpha=0.01, s=1, color='red', label='Data')
x_plot = np.unique(x)
plt.plot(x_plot, a_ols * x_plot + b_ols, 'k--', label='OLS')
plt.plot(x_plot, a_r_curr * x_plot + b_r_curr, 'b-', label='Current Robust')
plt.plot(x_plot, a_clip * x_plot + b_clip, 'g-', label='Proposed Clip Fit')
plt.legend()
plt.title("Background Fit Comparison")
plt.savefig('final_fit_analysis.png')
print("Analysis saved to final_fit_analysis.png")
