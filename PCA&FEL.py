import numpy as np
import matplotlib.pyplot as plt

PROJ_FILE = "pca_projection.dat"
OUT_SCATTER = "PCA_PC1_PC2_scatter.png"
OUT_FEL_2D = "FEL_PC1_PC2_2D.png"
OUT_FEL_3D = "FEL_PC1_PC2_3D.png"

T = 310.0
kB = 0.0019872041
kBT = kB * T
FRAME_MIN = 1
FRAME_MAX = 500
BINS = 30
CMAP = "viridis" 
SMOOTH_SIGMA = 1.2

def load_projection(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            frame = int(float(parts[0]))
            pc1 = float(parts[1])
            pc2 = float(parts[2])
            data.append((frame, pc1, pc2))
    arr = np.array(data, dtype=float)
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]
frames, pc1_all, pc2_all = load_projection(PROJ_FILE)

mask = (frames >= FRAME_MIN) & (frames <= FRAME_MAX)
pc1 = pc1_all[mask]
pc2 = pc2_all[mask]

if pc1.size < 50:
    raise RuntimeError(f"lack enough frames：{pc1.size}")


plt.figure(figsize=(6.5, 5.2), dpi=300)
plt.scatter(pc1, pc2, s=10, alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(OUT_SCATTER)
plt.close()


def gaussian_kernel_2d(sigma, radius=None):
    if sigma <= 0:
        return None
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    ker = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    ker /= ker.sum()
    return ker

def convolve2d_same(a, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    ap = np.pad(a, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(a, dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            patch = ap[i:i+kh, j:j+kw]
            out[i, j] = np.sum(patch * k)
    return out

H, xedges, yedges = np.histogram2d(pc1, pc2, bins=BINS, density=False)
if SMOOTH_SIGMA > 0:
    K = gaussian_kernel_2d(SMOOTH_SIGMA)
    Hs = convolve2d_same(H.astype(float), K)
else:
    Hs = H.astype(float)
P = Hs / np.sum(Hs)
P[P <= 0] = np.nan
F = -kBT * np.log(P)
F = F - np.nanmin(F)
xcent = 0.5 * (xedges[:-1] + xedges[1:])
ycent = 0.5 * (yedges[:-1] + yedges[1:])
X, Y = np.meshgrid(xcent, ycent, indexing="ij")


# FEL 2D
plt.figure(figsize=(6.5, 5.2), dpi=600)
cf = plt.contourf(X, Y, F, levels=40, cmap=CMAP)
plt.xlabel("PC1")
plt.ylabel("PC2")
cbar = plt.colorbar(cf)
cbar.set_label("ΔG (kcal/mol)")
plt.tight_layout()
plt.savefig(OUT_FEL_2D)
plt.savefig(OUT_FEL_2D.replace(".png", ".tiff"), dpi=600)
plt.close()

# FEL 3D surface
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(7.0, 5.5), dpi=600)
ax = fig.add_subplot(111, projection="3d")
F_plot = np.where(np.isfinite(F), F, np.nanmax(F[np.isfinite(F)]) + 1.0)
surf = ax.plot_surface(X, Y, F_plot, cmap=CMAP, linewidth=0, antialiased=True)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("ΔG (kcal/mol)")
fig.colorbar(surf, shrink=0.6, pad=0.1, label="ΔG (kcal/mol)")
plt.tight_layout()
plt.savefig(OUT_FEL_3D)
plt.savefig(OUT_FEL_3D.replace(".png", ".tiff"), dpi=600
)
plt.close()

print("Done:")
print(" -", OUT_SCATTER)
print(" -", OUT_FEL_2D)
print(" -", OUT_FEL_3D)
