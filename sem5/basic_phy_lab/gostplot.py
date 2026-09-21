"""gostplot — ГОСТ-styled plotting & approximation for lab practicums.

One-liner:
    GostPlot(x, y).xlabel('R', 'Ом').ylabel('U', 'В').fit(2).show()

Fluent builder. A `Dresser` holds the ГОСТ style underneath with sane
defaults; students never touch it unless they want to.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------- #
#  Dresser — pure style, knows nothing about data                             #
# --------------------------------------------------------------------------- #
@dataclass
class Dresser:
    """ГОСТ-ish visual style. Tweak fields or subclass for a different look."""

    # typography
    font_family: str = "serif"
    font_size: float = 13.0

    # data points
    marker: str = "o"
    marker_size: float = 5.0
    marker_face: str = "white"
    marker_edge: str = "black"

    # fit curve
    fit_color: str = "black"
    fit_width: float = 1.3

    # axes / grid (ГОСТ: full grid, minor ticks, ticks pointing inward)
    grid_major: str = "-"
    grid_minor: str = ":"
    grid_color: str = "0.6"
    tick_direction: str = "in"
    spine_width: float = 1.1

    figsize: tuple[float, float] = (7.0, 5.0)
    dpi: int = 120

    def apply(self, ax) -> None:
        plt.rcParams["font.family"] = self.font_family
        ax.tick_params(direction=self.tick_direction, which="both",
                       top=True, right=True)
        ax.minorticks_on()
        ax.grid(which="major", linestyle=self.grid_major,
                color=self.grid_color, linewidth=0.6)
        ax.grid(which="minor", linestyle=self.grid_minor,
                color=self.grid_color, linewidth=0.4, alpha=0.7)
        for s in ax.spines.values():
            s.set_linewidth(self.spine_width)

    @staticmethod
    def axis_label(name: str, unit: str | None) -> str:
        """ГОСТ label: «величина, единица» — e.g. ('R', 'Ом') -> 'R, Ом'."""
        return f"{name}, {unit}" if unit else name


# --------------------------------------------------------------------------- #
#  Fit result                                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class _Fit:
    xs: np.ndarray
    ys: np.ndarray
    equation: str
    r2: float
    params: np.ndarray          # best-fit values
    perr: np.ndarray            # 1σ uncertainty of each parameter
    names: list[str]            # parameter names (for the report line)
    weighted: bool              # True if dy was used (absolute_sigma)
    cov: np.ndarray             # full covariance matrix of the parameters
    model: Callable             # f(x, *params) -> y

    @property
    def report(self) -> str:
        """Multi-line 'name = value ± error' block."""
        return "\n".join(
            f"{n} = {v:.4g} ± {e:.2g}"
            for n, v, e in zip(self.names, self.params, self.perr)
        )


def _r_squared(y, y_pred) -> float:
    y = np.asarray(y, float)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def _poly_equation(coeffs: np.ndarray) -> str:
    # coeffs high-degree first (np.polyfit order)
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        p = n - i
        if p == 0:
            terms.append(f"{c:.4g}")
        elif p == 1:
            terms.append(f"{c:.4g}·x")
        else:
            terms.append(f"{c:.4g}·x^{p}")
    s = " + ".join(terms).replace("+ -", "− ")
    return f"y = {s}"


def _poly_names(deg: int) -> list[str]:
    # np.polyfit order: highest degree first
    return [f"a{deg - i}" for i in range(deg + 1)]


def _func_names(func: Callable) -> list[str]:
    # every arg after x
    return list(inspect.signature(func).parameters)[1:]


# --------------------------------------------------------------------------- #
#  GostPlot — fluent builder                                                  #
# --------------------------------------------------------------------------- #
class GostPlot:
    """Fluent ГОСТ plotter.

    Example
    -------
    >>> GostPlot(x, y).xlabel('R', 'Ом').ylabel('U', 'В').fit(2).show()
    """

    def __init__(self, x: Sequence[float], y: Sequence[float],
                 dresser: Dresser | None = None):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.dx = None          # x error bars (scalar or array)
        self.dy = None          # y error bars (scalar or array)
        self.dresser = dresser or Dresser()
        self._xlabel = ("x", None)
        self._ylabel = ("y", None)
        self._title: str | None = None
        self._fit: _Fit | None = None
        self._show_eq = True
        self._show_points = True
        self._pdf_path: str | None = None
        self._data_label = "эксперимент"
        self._fit_label = "аппроксимация"

    # --- labels ----------------------------------------------------------- #
    def xlabel(self, name: str, unit: str | None = None) -> "GostPlot":
        self._xlabel = (name, unit)
        return self

    def ylabel(self, name: str, unit: str | None = None) -> "GostPlot":
        self._ylabel = (name, unit)
        return self

    def title(self, text: str) -> "GostPlot":
        self._title = text
        return self

    def legend(self, data: str | None = None, fit: str | None = None) -> "GostPlot":
        if data is not None:
            self._data_label = data
        if fit is not None:
            self._fit_label = fit
        return self

    def errors(self, dy=None, dx=None) -> "GostPlot":
        """Measurement uncertainties for the data points.

        Each may be a scalar (same for all points) or an array (per point).
        `dy` also weights the fit (weighted least squares) when you call
        .fit() afterwards.
        """
        self.dy = None if dy is None else np.broadcast_to(dy, self.y.shape).astype(float)
        self.dx = None if dx is None else np.broadcast_to(dx, self.x.shape).astype(float)
        return self

    def style(self, dresser: Dresser) -> "GostPlot":
        self.dresser = dresser
        return self

    def show_equation(self, flag: bool = True) -> "GostPlot":
        self._show_eq = flag
        return self

    def save2pdf(self, path: bool | str = True) -> "GostPlot":
        """Also write a vector PDF when the plot is shown/rendered.

        .save2pdf(True)          -> filename from the title (or 'plot.pdf')
        .save2pdf('curve.pdf')   -> that exact path
        .save2pdf(False)         -> off
        """
        if path in (False, None):
            self._pdf_path = None
        elif path is True:
            base = self._title or "plot"
            slug = "".join(c if c.isalnum() or c in " -_" else "" for c in base)
            self._pdf_path = slug.strip().replace(" ", "_") + ".pdf"
        else:
            p = str(path)
            self._pdf_path = p if p.lower().endswith(".pdf") else p + ".pdf"
        return self

    def points(self, show: bool = True) -> "GostPlot":
        """Show/hide the experimental points (and their error bars).

        .points(False) leaves only the approximation curve — handy for a
        clean calibration line.
        """
        self._show_points = show
        return self

    # --- fitting ---------------------------------------------------------- #
    def fit(self, approx: int | Callable | None = 0, *,
            p0: Sequence[float] | None = None, n: int = 200) -> "GostPlot":
        """Approximate the data.

        approx = 0/None   -> no fit (points only)
        approx = int      -> polynomial of that degree (numpy.polyfit)
        approx = callable -> f(x, *params) fitted with scipy.curve_fit
        """
        if not approx:                       # 0, None, False
            self._fit = None
            return self

        xs = np.linspace(self.x.min(), self.x.max(), n)
        weighted = self.dy is not None       # -> use stated errors as absolute

        if callable(approx):
            # weighted least squares; absolute_sigma=True makes parameter
            # errors come from your dy, not just from residual scatter.
            params, cov = curve_fit(approx, self.x, self.y, p0=p0,
                                    sigma=self.dy, absolute_sigma=weighted)
            ys = approx(xs, *params)
            r2 = _r_squared(self.y, approx(self.x, *params))
            names = _func_names(approx)
            pretty = ", ".join(f"{p:.4g}" for p in params)
            eq = f"y = {approx.__name__}(x; {pretty})"
            model = approx

        elif isinstance(approx, int):
            w = None if self.dy is None else 1.0 / self.dy   # polyfit weight = 1/σ
            params, cov = np.polyfit(self.x, self.y, approx, w=w,
                                     cov=("unscaled" if weighted else True))
            ys = np.polyval(params, xs)
            r2 = _r_squared(self.y, np.polyval(params, self.x))
            names = _poly_names(approx)
            eq = _poly_equation(params)
            model = lambda x, *c: np.polyval(np.asarray(c), x)
        else:
            raise TypeError("approx must be 0/None, an int degree, or a callable")

        perr = np.sqrt(np.diag(cov))
        self._fit = _Fit(xs, ys, eq, r2, params, perr, names, weighted, cov, model)
        return self

    # --- prediction with propagated uncertainty --------------------------- #
    def predict(self, x, dx: float = 0.0):
        """Convert x → y through the fit, with propagated 1σ uncertainty.

        Combines two sources:
          * input error dx           -> (dy/dx · dx)²
          * fit-parameter covariance -> Jᵀ·Cov·J   (J = ∂y/∂params)
        Returns (y, sigma_y); both scalars or arrays matching x.
        Works for any model (polynomial or custom).
        """
        if self._fit is None:
            raise RuntimeError("call .fit(...) before .predict(...)")
        f, p, cov = self._fit.model, self._fit.params, self._fit.cov
        x = np.asarray(x, float)
        y = f(x, *p)

        # ∂y/∂params  (central differences, per parameter)
        J = np.empty(p.shape + x.shape) if x.shape else np.empty(p.shape)
        for i in range(len(p)):
            h = max(abs(p[i]) * 1e-6, 1e-9)
            pp, pm = p.copy(), p.copy()
            pp[i] += h; pm[i] -= h
            J[i] = (f(x, *pp) - f(x, *pm)) / (2 * h)
        var_par = np.einsum("i...,ij,j...->...", J, cov, J)

        # ∂y/∂x for the input-error term
        hx = np.maximum(np.abs(x) * 1e-6, 1e-9)
        dydx = (f(x + hx, *p) - f(x - hx, *p)) / (2 * hx)
        var_in = (dydx * dx) ** 2

        return y, np.sqrt(var_par + var_in)

    # --- render ----------------------------------------------------------- #
    def _render(self):
        d = self.dresser
        fig, ax = plt.subplots(figsize=d.figsize, dpi=d.dpi)
        if self._show_points:
            if self.dx is not None or self.dy is not None:
                ax.errorbar(self.x, self.y, yerr=self.dy, xerr=self.dx,
                            linestyle="none", marker=d.marker,
                            markersize=d.marker_size, markerfacecolor=d.marker_face,
                            markeredgecolor=d.marker_edge, ecolor=d.marker_edge,
                            elinewidth=0.9, capsize=3, label=self._data_label)
            else:
                ax.plot(self.x, self.y, linestyle="none", marker=d.marker,
                        markersize=d.marker_size, markerfacecolor=d.marker_face,
                        markeredgecolor=d.marker_edge, label=self._data_label)

        if self._fit is not None:
            lbl = self._fit_label
            if self._show_eq:
                lbl += (f"\n{self._fit.equation}"
                        f"\n{self._fit.report}"
                        f"\n$R^2$ = {self._fit.r2:.4f}")
            ax.plot(self._fit.xs, self._fit.ys, color=d.fit_color,
                    linewidth=d.fit_width, label=lbl)

        ax.set_xlabel(d.axis_label(*self._xlabel), fontsize=d.font_size)
        ax.set_ylabel(d.axis_label(*self._ylabel), fontsize=d.font_size)
        if self._title:
            ax.set_title(self._title, fontsize=d.font_size + 1)
        d.apply(ax)
        ax.legend(fontsize=d.font_size - 2, framealpha=0.9)
        fig.tight_layout()
        return fig, ax

    def show(self):
        fig, _ = self._render()
        if self._pdf_path:
            fig.savefig(self._pdf_path, bbox_inches="tight")
            print(f"[gostplot] сохранено: {self._pdf_path}")
        plt.show()
        return self

    def save(self, path: str, **kw):
        fig, _ = self._render()
        fig.savefig(path, bbox_inches="tight", **kw)
        plt.close(fig)
        return self

    @property
    def result(self) -> _Fit | None:
        """Access fit params / R² programmatically after .fit()."""
        return self._fit
