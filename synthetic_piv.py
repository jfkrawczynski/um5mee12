"""
synthetic_piv.py
-----------------
Lightweight synthetic PIV image generator for the "Practice" notebooks
(NB4 - Processing, NB5 - Analysis) of the UM5MEE12 PIV course.

This is a *pedagogical* stand-in for real aquarium acquisitions. It produces
grayscale double-frame particle images for three simplified flow models:

    - "free"      : uniform stream with mild broadband fluctuations
    - "cylinder"   : uniform stream + Gaussian wake deficit + oscillating
                     transverse component mimicking vortex shedding
                     (frequency set from a target Strouhal number)
    - "step"       : uniform stream + a simplified recirculation bubble
                     downstream of a backward-facing step

Replace the calls to `render_pair` / `render_sequence` with real image
loading (openpiv.tools.imread) once you have your own aquarium images -
the rest of the processing/analysis notebooks does not need to change.
"""
import numpy as np

# ---------------------------------------------------------------- geometry --
IMG_SHAPE = (192, 384)          # (ny, nx) pixels
CAL = 2.56                      # pixels / mm  (i.e. ~150 x 75 mm field of view)
FOV_MM = (IMG_SHAPE[1] / CAL, IMG_SHAPE[0] / CAL)   # (width, height) in mm

CYL_D_MM = 12.0                 # cylinder diameter, mm
CYL_X_MM, CYL_Y_MM = 35.0, FOV_MM[1] / 2            # cylinder centre, mm

STEP_H_MM = 15.0                # step height, mm
STEP_X_MM = 30.0                # streamwise location of the step, mm
STEP_LR_MM = 7.0 * STEP_H_MM    # recirculation length (~7h), mm

REGIMES = {"low": 0.6, "mid": 1.0, "high": 1.5}     # multiplies U0_REF
U0_REF_MM_S = 60.0              # reference free-stream velocity, mm/s
ST_TARGET = 0.20                # target Strouhal number for the cylinder wake
NU_WATER_MM2_S = 1.0            # kinematic viscosity of water, mm^2/s

FLOWS = ("free", "cylinder", "step")


def reynolds_number(flow, regime):
    """Nominal Reynolds number for reporting purposes."""
    U0 = U0_REF_MM_S * REGIMES[regime]
    L = CYL_D_MM if flow == "cylinder" else STEP_H_MM
    return U0 * L / NU_WATER_MM2_S


def shedding_frequency(regime):
    U0 = U0_REF_MM_S * REGIMES[regime]
    return ST_TARGET * U0 / CYL_D_MM  # Hz


# ------------------------------------------------------------- velocity field --
def velocity_field(flow, regime, x, y, t=0.0):
    """
    Return (u, v) in mm/s at physical coordinates x, y (mm), time t (s).
    x, y may be numpy arrays (broadcastable).
    """
    U0 = U0_REF_MM_S * REGIMES[regime]

    if flow == "free":
        # mild broadband fluctuation to look less "laminar-perfect"
        u = U0 + 0.03 * U0 * np.sin(0.4 * x + 3.1 * t) * np.cos(0.3 * y)
        v = 0.03 * U0 * np.cos(0.5 * x - 2.0 * t) * np.sin(0.4 * y)
        return u, v

    if flow == "cylinder":
        dx = x - CYL_X_MM
        dy = y - CYL_Y_MM
        r = np.sqrt(dx**2 + dy**2)
        u = np.full_like(x, U0, dtype=float)
        v = np.zeros_like(x, dtype=float)

        downstream = dx > CYL_D_MM / 2
        xi = np.clip(dx, 1e-3, None)               # avoid /0 upstream
        wake_width = 0.35 * CYL_D_MM * np.sqrt(xi / CYL_D_MM + 1.0)
        deficit_amp = 0.9 * U0 * (CYL_D_MM / (xi + CYL_D_MM)) ** 0.5
        deficit = deficit_amp * np.exp(-0.5 * (dy / wake_width) ** 2)

        f_shed = shedding_frequency(regime)
        k = 2 * np.pi * f_shed / (0.85 * U0)        # convective wavenumber
        shed_amp = 0.30 * U0 * np.exp(-xi / (10 * CYL_D_MM))
        v_shed = shed_amp * np.sin(2 * np.pi * f_shed * t - k * xi) * \
            np.exp(-0.5 * (dy / (0.6 * wake_width)) ** 2)

        u = np.where(downstream, u - deficit, u)
        v = np.where(downstream, v + v_shed, v)
        return u, v

    if flow == "step":
        dx = x - STEP_X_MM
        u = np.full_like(x, U0, dtype=float)
        v = np.zeros_like(x, dtype=float)

        in_bubble = (dx > 0) & (dx < STEP_LR_MM) & (y < STEP_H_MM)
        # normalized streamwise position in the bubble (0 at step, 1 at reattachment)
        xn = np.clip(dx / STEP_LR_MM, 1e-3, 1.0)
        # normalized wall-normal position in the shear layer (0 wall, 1 shear-layer edge)
        yn = np.clip(y / STEP_H_MM, 0.0, 1.0)
        recirc_u = -0.5 * U0 * np.sin(np.pi * xn) * np.sin(np.pi * yn / 2)
        u = np.where(in_bubble, recirc_u, u)

        # smooth shear-layer transition just above the bubble
        shear = (y >= STEP_H_MM) & (y < 1.6 * STEP_H_MM) & (dx > 0) & (dx < STEP_LR_MM)
        blend = (y - STEP_H_MM) / (0.6 * STEP_H_MM)
        u = np.where(shear, U0 * (0.15 + 0.85 * np.clip(blend, 0, 1)), u)
        return u, v

    raise ValueError(flow)


def solid_mask(flow, x, y):
    """Boolean array, True where a solid obstacle/wall blocks the flow."""
    if flow == "cylinder":
        return (x - CYL_X_MM) ** 2 + (y - CYL_Y_MM) ** 2 < (CYL_D_MM / 2) ** 2
    if flow == "step":
        return (x < STEP_X_MM) & (y < STEP_H_MM)
    return np.zeros_like(x, dtype=bool)


# ------------------------------------------------------------- image rendering --
def _random_particles(rng, n_particles):
    xs = rng.uniform(0, FOV_MM[0], n_particles)
    ys = rng.uniform(0, FOV_MM[1], n_particles)
    return xs, ys


def _render(xs_px, ys_px, shape, particle_sigma_px, intensities, noise_std, rng):
    img = np.zeros(shape, dtype=float)
    ny, nx = shape
    r = int(np.ceil(4 * particle_sigma_px))
    for x, y, I in zip(xs_px, ys_px, intensities):
        ix, iy = int(round(x)), int(round(y))
        x0, x1 = max(0, ix - r), min(nx, ix + r + 1)
        y0, y1 = max(0, iy - r), min(ny, iy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        img[y0:y1, x0:x1] += I * np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2 * particle_sigma_px ** 2)
        )
    img += rng.normal(0, noise_std, size=shape)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def render_pair(flow, regime, t=0.0, dt=0.06, n_particles=900, seed=None):
    """
    Generate one synthetic double-frame image pair.

    Returns
    -------
    frame_a, frame_b : uint8 ndarrays, shape IMG_SHAPE
    dt : float, the interframe time actually used (s)
    """
    rng = np.random.default_rng(seed)
    xs, ys = _random_particles(rng, n_particles)
    inside = solid_mask(flow, xs, ys)
    xs, ys = xs[~inside], ys[~inside]
    intensities = rng.normal(180, 35, size=xs.shape).clip(60, 255)
    sigma_px = rng.normal(1.1, 0.15, size=xs.shape).clip(0.6, 2.0)

    def to_px(x_mm, y_mm):
        return x_mm * CAL, y_mm * CAL

    xa_px, ya_px = to_px(xs, ys)
    frame_a = _render(xa_px, ya_px, IMG_SHAPE, sigma_px.mean(), intensities, 4.0, rng)

    u, v = velocity_field(flow, regime, xs, ys, t=t)
    xs_b = xs + u * dt
    ys_b = ys + v * dt
    keep = (
        (xs_b >= 0) & (xs_b < FOV_MM[0]) &
        (ys_b >= 0) & (ys_b < FOV_MM[1]) &
        (~solid_mask(flow, xs_b, ys_b))
    )
    xb_px, yb_px = to_px(xs_b[keep], ys_b[keep])
    frame_b = _render(xb_px, yb_px, IMG_SHAPE, sigma_px.mean(), intensities[keep], 4.0, rng)
    return frame_a, frame_b, dt


def render_sequence(flow, regime, n_frames=30, dt_frame=0.01, n_particles=900, seed=0):
    """
    Generate a time-resolved sequence of single frames (for time-series /
    spectral analysis, e.g. vortex-shedding frequency). Consecutive frames
    are meant to be cross-correlated as PIV pairs with dt = dt_frame.
    """
    rng = np.random.default_rng(seed)
    xs, ys = _random_particles(rng, n_particles)
    inside = solid_mask(flow, xs, ys)
    xs, ys = xs[~inside], ys[~inside]
    intensities = rng.normal(180, 35, size=xs.shape).clip(60, 255)
    sigma_px = 1.1

    frames = []
    x, y = xs.copy(), ys.copy()
    for i in range(n_frames):
        t = i * dt_frame
        xpx, ypx = x * CAL, y * CAL
        frames.append(_render(xpx, ypx, IMG_SHAPE, sigma_px, intensities, 4.0, rng))
        u, v = velocity_field(flow, regime, x, y, t=t)
        x = x + u * dt_frame
        y = y + v * dt_frame
        out = (x < 0) | (x >= FOV_MM[0]) | (y < 0) | (y >= FOV_MM[1]) | solid_mask(flow, x, y)
        if out.any():
            xn, yn = _random_particles(rng, int(out.sum()))
            x[out], y[out] = xn, yn
            intensities[out] = rng.normal(180, 35, size=int(out.sum())).clip(60, 255)
    return frames, dt_frame
