"""
truth_model.py  --  Task 1: Generate true aircraft motion.

Constant-velocity motion in 2D:
    x_k = x0 + vx * T * k
    y_k = y0 + vy * T * k

This is the "ground truth" the rest of the pipeline is evaluated against.
Run this file directly to see the truth trajectory on its own.
"""

import numpy as np
import matplotlib.pyplot as plt
import config as cfg


def generate_truth():
    """Returns true (x, y, vx, vy) arrays of length N."""
    k = np.arange(cfg.N)
    x_true = cfg.X0 + cfg.VX_TRUE * cfg.T * k
    y_true = cfg.Y0 + cfg.VY_TRUE * cfg.T * k
    vx_true = np.full(cfg.N, cfg.VX_TRUE)
    vy_true = np.full(cfg.N, cfg.VY_TRUE)
    return x_true, y_true, vx_true, vy_true


def plot_truth(x_true, y_true, save_path=None):
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 5.5))

    for ax in (ax_full, ax_zoom):
        ax.plot(x_true, y_true, 'g-', linewidth=2, label='True trajectory')
        ax.grid(True, alpha=0.3)

    # Left: full geometry (radar + trajectory), equal aspect
    ax_full.scatter([0], [0], marker='^', c='k', s=100, label='Radar (origin)')
    ax_full.set_xlabel('x (m)'); ax_full.set_ylabel('y (m)')
    ax_full.set_title('Full geometry (radar to target)')
    ax_full.legend(loc='lower right', fontsize=9)
    ax_full.set_aspect('equal')

    # Right: zoomed-in trajectory detail
    margin = 0.08 * (x_true.max() - x_true.min())
    ax_zoom.plot(x_true, y_true, 'g.', markersize=4)
    ax_zoom.set_xlim(x_true.min() - margin, x_true.max() + margin)
    ax_zoom.set_ylim(y_true.min() - margin, y_true.max() + margin)
    ax_zoom.set_xlabel('x (m)'); ax_zoom.set_ylabel('y (m)')
    ax_zoom.set_title('Zoomed: trajectory detail')

    fig.suptitle('Task 1: True aircraft trajectory (constant velocity)', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved {save_path}")
    plt.close()


if __name__ == "__main__":
    x_true, y_true, vx_true, vy_true = generate_truth()
    plot_truth(x_true, y_true, save_path="outputs/01_truth_trajectory.png")
    print(f"Trajectory spans x: [{x_true.min():.0f}, {x_true.max():.0f}] m, "
          f"y: [{y_true.min():.0f}, {y_true.max():.0f}] m")
