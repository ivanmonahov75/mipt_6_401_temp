"""Demo: the three approximation modes."""
import numpy as np
from gostplot import GostPlot, Dresser

rng = np.random.default_rng(0)

# 1) no fit — points only
x = np.array([1, 2, 3, 4, 5, 6], float)
y = np.array([2.1, 3.9, 6.2, 8.1, 9.8, 12.2])
GostPlot(x, y).xlabel("I", "мА").ylabel("U", "В").fit(0).save("out_none.png")

# 2) polynomial degree 1 with error bars -> weighted fit + param errors
(GostPlot(x, y)
    .xlabel("I", "мА").ylabel("U", "В")
    .errors(dy=0.3, dx=0.1)      # scalar: same uncertainty on every point
    .fit(1)
    .save("out_poly1.png"))

# 3) custom function — exponential decay
xd = np.linspace(0, 5, 12)
yd = 4.0 * np.exp(-0.7 * xd) + 0.2 + rng.normal(0, 0.05, xd.size)

def decay(x, a, k, c):
    return a * np.exp(-k * x) + c

(GostPlot(xd, yd)
    .xlabel("t", "с").ylabel("A", "отн. ед.")
    .title("Затухание")
    .errors(dy=0.05)
    .fit(decay, p0=[4, 1, 0])
    .save("out_custom.png"))

# programmatic access to params + uncertainties
g = GostPlot(x, y).errors(dy=0.3).fit(1)
for name, val, err in zip(g.result.names, g.result.params, g.result.perr):
    print(f"  {name} = {val:.4g} ± {err:.2g}")
print("R2 =", round(g.result.r2, 4))
print("saved: out_none.png, out_poly1.png, out_custom.png")
