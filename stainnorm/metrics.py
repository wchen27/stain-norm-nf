import numpy as np

_M = np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb2lab(rgb):
    c = np.asarray(rgb, dtype=np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ _M.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(lab1, lab2):
    L1, a1, b1 = np.moveaxis(lab1, -1, 0)
    L2, a2, b2 = np.moveaxis(lab2, -1, 0)
    cbar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7)))
    a1p, a2p = a1 * (1 + g), a2 * (1 + g)
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.arctan2(b1, a1p) % (2 * np.pi)
    h2p = np.arctan2(b2, a2p) % (2 * np.pi)
    dL, dC = L2 - L1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > np.pi, dh - 2 * np.pi, np.where(dh < -np.pi, dh + 2 * np.pi, dh))
    dh = np.where(c1p * c2p == 0, 0, dh)
    dH = 2 * np.sqrt(c1p * c2p) * np.sin(dh / 2)
    Lbar, Cbar = (L1 + L2) / 2, (c1p + c2p) / 2
    hsum = h1p + h2p
    hbar = np.where(np.abs(h1p - h2p) > np.pi, (hsum + 2 * np.pi) / 2, hsum / 2)
    hbar = np.where(c1p * c2p == 0, hsum, hbar) % (2 * np.pi)
    t = (1 - 0.17 * np.cos(hbar - np.radians(30)) + 0.24 * np.cos(2 * hbar)
         + 0.32 * np.cos(3 * hbar + np.radians(6)) - 0.20 * np.cos(4 * hbar - np.radians(63)))
    sl = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    sc = 1 + 0.045 * Cbar
    sh = 1 + 0.015 * Cbar * t
    rt = -2 * np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25 ** 7)) * np.sin(np.radians(60) * np.exp(-((np.degrees(hbar) - 275) / 25) ** 2))
    return np.sqrt((dL / sl) ** 2 + (dC / sc) ** 2 + (dH / sh) ** 2 + rt * (dC / sc) * (dH / sh))


def delta_e2000(rgb1, rgb2, mask=None):
    d = ciede2000(rgb2lab(rgb1), rgb2lab(rgb2))
    return float(d[mask].mean() if mask is not None else d.mean())


def _box(x, w=7):
    c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1)
    return (c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]) / (w * w)


def ssim(rgb1, rgb2, w=7):
    g = np.array([0.2125, 0.7154, 0.0721])
    x, y = np.asarray(rgb1, np.float64) @ g, np.asarray(rgb2, np.float64) @ g
    n = w * w
    ux, uy = _box(x, w), _box(y, w)
    vx = (_box(x * x, w) - ux * ux) * n / (n - 1)
    vy = (_box(y * y, w) - uy * uy) * n / (n - 1)
    vxy = (_box(x * y, w) - ux * uy) * n / (n - 1)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def tissue_mask(rgb, beta=0.15, io=240):
    od = -np.log((np.asarray(rgb, np.float64) + 1) / io)
    return ~np.any(od < beta, axis=-1)
