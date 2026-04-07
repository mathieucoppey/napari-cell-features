import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
csv_path = r'c:\Users\coppey\napari-cell-features\scatter_491_s5_405_s5_frame45.csv'
df = pd.read_csv(csv_path)

# Extract background points (Label == 0)
bg_df = df[df['Label'] == 0]
bg_x = bg_df['X'].values
bg_y = bg_df['Y'].values

if len(bg_x) > 2:
    # OLS Fit
    a_bg, b_bg = np.polyfit(bg_x, bg_y, 1)
    print(f"Background OLS Fit: y = {a_bg:.6f}x + {b_bg:.2f}")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(bg_x, bg_y, alpha=0.1, s=1, color='red', label='Background')
    
    x_range = np.array([0, np.max(df['X'])])
    y_range = a_bg * x_range + b_bg
    plt.plot(x_range, y_range, 'k--', label=f'OLS: y = {a_bg:.3f}x + {b_bg:.1f}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Background Scatter and OLS Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = r'c:\Users\coppey\napari-cell-features\debug_fit_result.png'
    plt.savefig(output_path)
    print(f"Saved debug plot to {output_path}")
else:
    print("Not enough background points for fitting.")
