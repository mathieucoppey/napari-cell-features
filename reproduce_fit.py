import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def robust_linear_fit(x, y):
    if len(x) < 2:
        return 0.0, 0.0
    def model(params, x):
        return params[0] * x + params[1]
    def residuals(params, x, y):
        return model(params, x) - y
    try:
        x_stacked = np.vstack([x, np.ones(len(x))]).T
        a0, b0 = np.linalg.lstsq(x_stacked, y, rcond=None)[0]
    except Exception:
        a0, b0 = 1.0, 0.0
    res = least_squares(residuals, [a0, b0], args=(x, y), loss='soft_l1', f_scale=1000.0) # Adjusted f_scale
    return res.x

# Simulate User's Data
# Background cluster (Dense)
bg_n = 10000
bg_x = np.random.normal(8000, 1000, bg_n)
bg_y = np.random.normal(5000, 500, bg_n)

# Missed Cell points labeled as Background (Outliers)
outlier_n = 2000
outlier_x = np.random.normal(25000, 2000, outlier_n)
outlier_y = np.random.normal(25000, 2000, outlier_n)

# Combine points labeled as Background
all_bg_x = np.concatenate([bg_x, outlier_x])
all_bg_y = np.concatenate([bg_y, outlier_y])

# Current implementation OLS
a_ols, b_ols = np.polyfit(all_bg_x, all_bg_y, 1)

# Current implementation Robust (with f_scale=1.0)
def robust_linear_fit_f1(x, y):
    def model(params, x): return params[0] * x + params[1]
    def residuals(params, x, y): return model(params, x) - y
    a0, b0 = np.polyfit(x, y, 1)
    res = least_squares(residuals, [a0, b0], args=(x, y), loss='soft_l1', f_scale=1.0)
    return res.x

a_r1, b_r1 = robust_linear_fit_f1(all_bg_x, all_bg_y)

# Proposed: RANSAC or similar
def simple_ransac(x, y, iterations=100, threshold=1000):
    best_a, best_b = 0, 0
    max_inliers = -1
    for _ in range(iterations):
        indices = np.random.choice(len(x), 2, replace=False)
        x_s, y_s = x[indices], y[indices]
        if x_s[1] == x_s[0]: continue
        a = (y_s[1] - y_s[0]) / (x_s[1] - x_s[0])
        b = y_s[0] - a * x_s[0]
        residuals = np.abs(a * x + b - y)
        inliers = np.sum(residuals < threshold)
        if inliers > max_inliers:
            max_inliers = inliers
            best_a, best_b = a, b
    return best_a, best_b

a_ransac, b_ransac = simple_ransac(all_bg_x, all_bg_y)

print(f"OLS: y = {a_ols:.3f}x + {b_ols:.1f}")
print(f"Robust (f=1): y = {a_r1:.3f}x + {b_r1:.1f}")
print(f"RANSAC: y = {a_ransac:.3f}x + {b_ransac:.1f}")

plt.scatter(all_bg_x, all_bg_y, alpha=0.1, s=1, color='red')
x_range = np.array([0, 60000])
plt.plot(x_range, a_ols * x_range + b_ols, 'k--', label='OLS')
plt.plot(x_range, a_r1 * x_range + b_r1, 'b-', label='Robust (f=1)')
plt.plot(x_range, a_ransac * x_range + b_ransac, 'g-', label='RANSAC')
plt.legend()
plt.savefig('fit_comparison.png')
print("Saved comparison to fit_comparison.png")
