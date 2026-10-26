import tkinter as tk
from tkinter import ttk, messagebox, font
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")

BG_DARK       = "#0d1117"
BG_CARD       = "#161b22"
BG_INPUT      = "#1c2128"
BG_TABLE      = "#0d1117"
ACCENT_BLUE   = "#1f6feb"
ACCENT_LIGHT  = "#58a6ff"
ACCENT_GREEN  = "#3fb950"
ACCENT_RED    = "#f85149"
ACCENT_YELLOW = "#d29922"
TEXT_PRIMARY  = "#e6edf3"
TEXT_MUTED    = "#8b949e"
BORDER_COLOR  = "#30363d"
FONT_MONO     = ("Consolas", 10)
FONT_TITLE    = ("Segoe UI", 14, "bold")
FONT_LABEL    = ("Segoe UI", 10)
FONT_SMALL    = ("Segoe UI", 9)
FONT_BTN      = ("Segoe UI", 10, "bold")


def safe_eval(expr_str, x_val):
    """Safely evaluate a mathematical expression at a given x value."""
    x = sp.Symbol('x')
    try:
        expr = sp.sympify(expr_str)
        func = sp.lambdify(x, expr, modules=["numpy", "sympy"])
        result = func(x_val)
        return float(result)
    except Exception as e:
        raise ValueError(f"Error evaluating f({x_val}): {e}")

def get_sympy_expr(expr_str):
    """Parse expression string to sympy expression."""
    x = sp.Symbol('x')
    try:
        return sp.sympify(expr_str), x
    except Exception as e:
        raise ValueError(f"Invalid function expression: {e}")

# Bisection Method
def bisection_method(fx_str, a, b, tol, max_iter):
    rows = []
    fa = safe_eval(fx_str, a)
    fb = safe_eval(fx_str, b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for Bisection.")
    root = None
    error = float('inf')
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = safe_eval(fx_str, c)
        if i > 1:
            error = abs(c - prev_c) / abs(c) * 100 if c != 0 else abs(c - prev_c)
        else:
            error = float('nan')
        rows.append((i, a, b, c, fc, error if i > 1 else "---"))
        if abs(fc) < 1e-15 or (i > 1 and abs(error) < tol):
            root = c
            break
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        prev_c = c
    if root is None:
        root = (a + b) / 2.0
    final_error = abs(safe_eval(fx_str, root))
    return root, len(rows), final_error, rows

#  False Position Method
def false_position_method(fx_str, a, b, tol, max_iter):
    rows = []
    fa = safe_eval(fx_str, a)
    fb = safe_eval(fx_str, b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs for False Position.")
    root = None
    prev_c = None
    for i in range(1, max_iter + 1):
        fc_a = safe_eval(fx_str, a)
        fc_b = safe_eval(fx_str, b)
        if fc_b - fc_a == 0:
            raise ValueError("Division by zero in False Position method.")
        c = b - fc_b * (b - a) / (fc_b - fc_a)
        fc = safe_eval(fx_str, c)
        if prev_c is not None and c != 0:
            error = abs(c - prev_c) / abs(c) * 100
        else:
            error = float('nan')
        rows.append((i, a, b, c, fc, error if prev_c is not None else "---"))
        if prev_c is not None and abs(error) < tol:
            root = c
            break
        if fc_a * fc < 0:
            b = c
        else:
            a = c
        prev_c = c
    if root is None:
        root = c
    final_error = abs(safe_eval(fx_str, root))
    return root, len(rows), final_error, rows

# Newton-Raphson Method
def newton_raphson_method(fx_str, x0, tol, max_iter):
    expr, x_sym = get_sympy_expr(fx_str)
    df_expr = sp.diff(expr, x_sym)
    df_func = sp.lambdify(x_sym, df_expr, modules=["numpy", "sympy"])
    rows = []
    xi = float(x0)
    root = None
    for i in range(1, max_iter + 1):
        fxi = safe_eval(fx_str, xi)
        dfxi = float(df_func(xi))
        if abs(dfxi) < 1e-15:
            raise ValueError(f"Derivative is zero at x={xi}. Newton-Raphson fails.")
        xi_new = xi - fxi / dfxi
        error = abs(xi_new - xi) / abs(xi_new) * 100 if xi_new != 0 else abs(xi_new - xi)
        rows.append((i, xi, fxi, dfxi, xi_new, error))
        xi = xi_new
        if error < tol:
            root = xi
            break
    if root is None:
        root = xi
    final_error = abs(safe_eval(fx_str, root))
    return root, len(rows), final_error, rows

#  Secant Method
def secant_method(fx_str, x0, x1, tol, max_iter):
    rows = []
    xi_1 = float(x0)
    xi   = float(x1)
    root = None
    for i in range(1, max_iter + 1):
        fxi_1 = safe_eval(fx_str, xi_1)
        fxi   = safe_eval(fx_str, xi)
        denom = fxi - fxi_1
        if abs(denom) < 1e-15:
            raise ValueError("Division by zero in Secant Method.")
        xi_new = xi - fxi * (xi - xi_1) / denom
        error = abs(xi_new - xi) / abs(xi_new) * 100 if xi_new != 0 else abs(xi_new - xi)
        rows.append((i, xi_1, xi, fxi, xi_new, error))
        xi_1 = xi
        xi   = xi_new
        if error < tol:
            root = xi
            break
    if root is None:
        root = xi
    final_error = abs(safe_eval(fx_str, root))
    return root, len(rows), final_error, rows

# Gauss-Seidel Method
def gauss_seidel_method(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    rows = []
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            if abs(A[i][i]) < 1e-15:
                raise ValueError(f"Zero diagonal element at row {i}. System may not converge.")
            x[i] = (b[i] - sigma) / A[i][i]
        error = np.linalg.norm(x - x_old, np.inf)
        rows.append((iteration, x.copy(), error))
        if error < tol:
            break
    return x, len(rows), rows

#  Jacobi Method
def jacobi_method(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    rows = []
    for iteration in range(1, max_iter + 1):
        x_new = np.zeros(n)
        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            if abs(A[i][i]) < 1e-15:
                raise ValueError(f"Zero diagonal element at row {i}. System may not converge.")
            x_new[i] = (b[i] - sigma) / A[i][i]
        error = np.linalg.norm(x_new - x, np.inf)
        rows.append((iteration, x_new.copy(), error))
        x = x_new.copy()
        if error < tol:
            break
    return x, len(rows), rows


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ── Interpolation helper
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_float_list(s):
    """Parse a comma-separated string into a list of floats."""
    try:
        return [float(v.strip()) for v in s.split(",") if v.strip()]
    except Exception:
        raise ValueError("Could not parse values. Use comma-separated numbers (e.g. 1, 2, 3).")


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ── Newton Forward Interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def newton_forward_interp(x_vals, y_vals, x_target):
    """Newton's Forward Difference Interpolation (equal spacing)."""
    n = len(x_vals)
    if n < 2:
        raise ValueError("At least 2 data points are required.")
    h = x_vals[1] - x_vals[0]
    if abs(h) < 1e-15:
        raise ValueError("x values must be distinct.")
    for i in range(1, n - 1):
        if not np.isclose(x_vals[i + 1] - x_vals[i], h, rtol=1e-4):
            raise ValueError("Newton Forward requires equally-spaced x values.")

    # Build forward difference table
    diff = np.zeros((n, n))
    diff[:, 0] = np.array(y_vals, dtype=float)
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    s = (x_target - x_vals[0]) / h
    result = diff[0][0]
    s_term = 1.0
    factorial = 1.0
    terms = [(0, diff[0][0], 1.0, diff[0][0])]   # (order, delta, coeff, contribution)
    for k in range(1, n):
        s_term *= (s - (k - 1))
        factorial *= k
        coeff = s_term / factorial
        contrib = coeff * diff[0][k]
        result += contrib
        terms.append((k, diff[0][k], coeff, contrib))

    return result, diff, s, terms


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ── Newton Backward Interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def newton_backward_interp(x_vals, y_vals, x_target):
    """Newton's Backward Difference Interpolation (equal spacing)."""
    n = len(x_vals)
    if n < 2:
        raise ValueError("At least 2 data points are required.")
    h = x_vals[1] - x_vals[0]
    if abs(h) < 1e-15:
        raise ValueError("x values must be distinct.")
    for i in range(1, n - 1):
        if not np.isclose(x_vals[i + 1] - x_vals[i], h, rtol=1e-4):
            raise ValueError("Newton Backward requires equally-spaced x values.")

    # Build forward difference table (backward uses bottom row)
    diff = np.zeros((n, n))
    diff[:, 0] = np.array(y_vals, dtype=float)
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    s = (x_target - x_vals[-1]) / h
    result = diff[n - 1][0]
    s_term = 1.0
    factorial = 1.0
    terms = [(0, diff[n - 1][0], 1.0, diff[n - 1][0])]
    for k in range(1, n):
        s_term *= (s + (k - 1))
        factorial *= k
        idx = n - 1 - k
        if idx < 0:
            break
        coeff = s_term / factorial
        contrib = coeff * diff[idx][k]
        result += contrib
        terms.append((k, diff[idx][k], coeff, contrib))

    return result, diff, s, terms


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ── Stirling's Interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def stirling_interp(x_vals, y_vals, x_target):
    """Stirling's Central Difference Interpolation (equal spacing, central point)."""
    n = len(x_vals)
    if n < 3:
        raise ValueError("Stirling's formula requires at least 3 data points.")
    h = x_vals[1] - x_vals[0]
    if abs(h) < 1e-15:
        raise ValueError("x values must be distinct.")
    for i in range(1, n - 1):
        if not np.isclose(x_vals[i + 1] - x_vals[i], h, rtol=1e-4):
            raise ValueError("Stirling's formula requires equally-spaced x values.")

    # Build forward difference table
    diff = np.zeros((n, n))
    diff[:, 0] = np.array(y_vals, dtype=float)
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    mid = n // 2
    s = (x_target - x_vals[mid]) / h
    result = diff[mid][0]
    terms = [(0, diff[mid][0], "y₀", 1.0, diff[mid][0])]

    # 1st order: s · (Δy₋₁ + Δy₀) / 2  i.e. mean of diff[mid-1][1] and diff[mid][1]
    if mid >= 1 and n > 1:
        mu1 = (diff[mid][1] + diff[mid - 1][1]) / 2.0
        coeff1 = s
        v1 = coeff1 * mu1
        result += v1
        terms.append((1, mu1, "μδy₀", coeff1, v1))

    # 2nd order: s²/2! · δ²y₀ = diff[mid-1][2]
    if mid >= 1 and n > 2:
        d2 = diff[mid - 1][2]
        coeff2 = s ** 2 / 2.0
        v2 = coeff2 * d2
        result += v2
        terms.append((2, d2, "δ²y₀", coeff2, v2))

    # 3rd order: s(s²-1)/3! · (Δ³y₋₂ + Δ³y₋₁) / 2
    if mid >= 2 and n > 3:
        mu3 = (diff[mid - 1][3] + diff[mid - 2][3]) / 2.0
        coeff3 = s * (s ** 2 - 1) / 6.0
        v3 = coeff3 * mu3
        result += v3
        terms.append((3, mu3, "μδ³y₀", coeff3, v3))

    # 4th order: s²(s²-1)/4! · δ⁴y₀ = diff[mid-2][4]
    if mid >= 2 and n > 4:
        d4 = diff[mid - 2][4]
        coeff4 = s ** 2 * (s ** 2 - 1) / 24.0
        v4 = coeff4 * d4
        result += v4
        terms.append((4, d4, "δ⁴y₀", coeff4, v4))

    return result, diff, mid, s, terms


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ── Lagrange Interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def lagrange_interp(x_vals, y_vals, x_target):
    """Lagrange Interpolation (works for unequal spacing)."""
    n = len(x_vals)
    if n < 2:
        raise ValueError("At least 2 data points are required.")
    # Check for duplicate x values
    for i in range(n):
        for j in range(i + 1, n):
            if np.isclose(x_vals[i], x_vals[j]):
                raise ValueError(f"Duplicate x value detected at index {i} and {j}.")

    result = 0.0
    terms = []   # (k, xk, yk, Lk, contribution)
    for i in range(n):
        numerator = 1.0
        denominator = 1.0
        for j in range(n):
            if j != i:
                numerator   *= (x_target - x_vals[j])
                denominator *= (x_vals[i] - x_vals[j])
        Li = numerator / denominator
        contrib = Li * y_vals[i]
        result += contrib
        terms.append((i, x_vals[i], y_vals[i], Li, contrib))

    return result, terms


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ── Interpolation Plot Helper
# ═══════════════════════════════════════════════════════════════════════════════

def plot_interpolation(ax, x_vals, y_vals, x_target, y_interp, method_name, interp_callable):
    """Plot the interpolating polynomial, data points, and the interpolated value."""
    ax.clear()
    ax.set_facecolor(BG_DARK)
    span = max(x_vals) - min(x_vals)
    x_min = min(x_vals) - span * 0.15
    x_max = max(x_vals) + span * 0.15
    x_plot = np.linspace(x_min, x_max, 600)
    y_plot = []
    for xp in x_plot:
        try:
            yp = interp_callable(xp)
            y_plot.append(float(yp) if not (np.isnan(float(yp)) or np.isinf(float(yp))) else np.nan)
        except Exception:
            y_plot.append(np.nan)
    y_plot = np.array(y_plot)

    ax.plot(x_plot, y_plot, color=ACCENT_LIGHT, linewidth=2.2,
            label="Interpolating polynomial", zorder=3)
    ax.scatter(x_vals, y_vals, color=ACCENT_YELLOW, s=80, zorder=5,
               label="Data points", edgecolors=BG_DARK, linewidths=0.8)
    ax.scatter([x_target], [y_interp], color=ACCENT_GREEN, s=150, zorder=6,
               marker="*", label=f"f({x_target:.4g}) ≈ {y_interp:.6f}")
    ax.axvline(x_target, color=ACCENT_GREEN, linewidth=1.1, linestyle=":", alpha=0.6)
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.7, linestyle="--", alpha=0.4)

    finite_y = [v for v in y_plot if not np.isnan(v) and not np.isinf(v)]
    if finite_y:
        ymax = np.percentile(np.abs(finite_y), 96)
        ax.set_ylim(-ymax * 1.4, ymax * 1.4)

    ax.set_title(method_name, color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("x", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("f(x)", color=TEXT_MUTED, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.grid(True, color=BORDER_COLOR, linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY, fontsize=9)


def plot_root_function(ax, fx_str, root, method_name, a=None, b=None):
    """Plot f(x) with root highlighted."""
    ax.clear()
    ax.set_facecolor(BG_DARK)
    center = root if root is not None else 0
    margin = max(abs(a if a else center) * 2, abs(b if b else center) * 2, 3)
    x_min = center - margin
    x_max = center + margin
    x_vals = np.linspace(x_min, x_max, 800)
    y_vals = []
    expr, x_sym = get_sympy_expr(fx_str)
    f_lam = sp.lambdify(x_sym, expr, modules=["numpy"])
    try:
        y_vals = f_lam(x_vals)
        y_vals = np.where(np.abs(y_vals) > 1e10, np.nan, y_vals)
    except Exception:
        y_vals = np.full_like(x_vals, np.nan)
    ax.plot(x_vals, y_vals, color=ACCENT_LIGHT, linewidth=2.2, label=f"f(x) = {fx_str}", zorder=3)
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color=TEXT_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    if root is not None:
        froot = 0
        try:
            froot = safe_eval(fx_str, root)
        except:
            pass
        ax.scatter([root], [froot], color=ACCENT_GREEN, s=120, zorder=5, label=f"Root ≈ {root:.6f}")
        ax.axvline(root, color=ACCENT_GREEN, linewidth=1.2, linestyle=":", alpha=0.7)
    if a is not None and b is not None:
        ax.axvline(a, color=ACCENT_YELLOW, linewidth=1, linestyle="--", alpha=0.5, label=f"a={a}")
        ax.axvline(b, color=ACCENT_RED,    linewidth=1, linestyle="--", alpha=0.5, label=f"b={b}")
    y_data = [v for v in y_vals if not np.isnan(v) and not np.isinf(v)]
    if y_data:
        ymax = np.percentile(np.abs(y_data), 95)
        ax.set_ylim(-ymax * 1.4, ymax * 1.4)
    ax.set_title(f"{method_name}", color=TEXT_PRIMARY, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("x", color=TEXT_MUTED, fontsize=10)
    ax.set_ylabel("f(x)", color=TEXT_MUTED, fontsize=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    ax.grid(True, color=BORDER_COLOR, linewidth=0.5, alpha=0.5)
    ax.legend(facecolor=BG_CARD, edgecolor=BORDER_COLOR, labelcolor=TEXT_PRIMARY, fontsize=9)


class NumericalCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Numerical Methods Calculator")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x800")
        self.root.minsize(1100, 700)
        self._setup_styles()
        self._build_header()
        self._build_main_layout()
        self._build_method_selector()
        self.method_frames = {}
        self._build_bisection_frame()
        self._build_false_position_frame()
        self._build_newton_raphson_frame()
        self._build_secant_frame()
        self._build_gauss_seidel_frame()
        self._build_jacobi_frame()
        # ── NEW interpolation frames ───────────────────────────────────────────
        self._build_newton_forward_frame()
        self._build_newton_backward_frame()
        self._build_stirling_frame()
        self._build_lagrange_frame()
        # ──────────────────────────────────────────────────────────────────────
        self.current_method = None
        self.show_method("Bisection")

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Dark.TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("Dark.TLabel", background=BG_DARK, foreground=TEXT_PRIMARY, font=FONT_LABEL)
        style.configure("Card.TLabel", background=BG_CARD, foreground=TEXT_PRIMARY, font=FONT_LABEL)
        style.configure("Muted.TLabel", background=BG_CARD, foreground=TEXT_MUTED, font=FONT_SMALL)
        style.configure("Title.TLabel", background=BG_DARK, foreground=TEXT_PRIMARY, font=FONT_TITLE)
        style.configure("Treeview",
            background=BG_TABLE, foreground=TEXT_PRIMARY,
            fieldbackground=BG_TABLE, borderwidth=0,
            rowheight=22, font=FONT_MONO)
        style.configure("Treeview.Heading",
            background=BG_CARD, foreground=ACCENT_LIGHT,
            font=("Segoe UI", 9, "bold"), borderwidth=0)
        style.map("Treeview",
            background=[("selected", ACCENT_BLUE)],
            foreground=[("selected", TEXT_PRIMARY)])
        style.configure("Run.TButton",
            background=ACCENT_BLUE, foreground=TEXT_PRIMARY,
            font=FONT_BTN, borderwidth=0, padding=(14, 7))
        style.map("Run.TButton",
            background=[("active", "#388bfd"), ("pressed", "#1158c7")])
        style.configure("Reset.TButton",
            background=BG_INPUT, foreground=TEXT_MUTED,
            font=FONT_BTN, borderwidth=0, padding=(14, 7))
        style.map("Reset.TButton",
            background=[("active", "#2d333b"), ("pressed", "#1c2128")])

    def _build_header(self):
        hdr = tk.Frame(self.root, bg="#161b22", height=58)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚙  Advanced Numerical Methods Calculator",
                 bg="#161b22", fg=TEXT_PRIMARY,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=22, pady=12)
        tk.Label(hdr, text="Academic GUI Tool · Tkinter + Matplotlib + Sympy",
                 bg="#161b22", fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="right", padx=22, pady=18)
        sep = tk.Frame(self.root, bg=BORDER_COLOR, height=1)
        sep.pack(fill="x")

    def _build_main_layout(self):
        self.main_pane = tk.PanedWindow(self.root, orient="horizontal",
                                        bg=BG_DARK, sashwidth=4,
                                        sashrelief="flat", sashpad=0)
        self.main_pane.pack(fill="both", expand=True, padx=0, pady=0)
        # Left sidebar
        self.sidebar = tk.Frame(self.main_pane, bg=BG_CARD, width=210)
        self.main_pane.add(self.sidebar, minsize=180)
        # Right content
        self.content = tk.Frame(self.main_pane, bg=BG_DARK)
        self.main_pane.add(self.content, minsize=700)

    def _build_method_selector(self):
        tk.Label(self.sidebar, text="METHODS", bg=BG_CARD,
                 fg=TEXT_MUTED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(18, 6))
        tk.Frame(self.sidebar, bg=BORDER_COLOR, height=1).pack(fill="x", padx=12)
        self.method_groups = {
            "Root Finding":    ["Bisection", "False Position", "Newton-Raphson", "Secant"],
            "Linear Systems":  ["Gauss-Seidel", "Jacobi"],
            # ── NEW ──────────────────────────────────────────────────────────
            "Interpolation":   ["Newton Forward", "Newton Backward", "Stirling", "Lagrange"],
        }
        self.method_btns = {}
        for group, methods in self.method_groups.items():
            tk.Label(self.sidebar, text=group.upper(), bg=BG_CARD,
                     fg=ACCENT_LIGHT, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=16, pady=(14, 4))
            for m in methods:
                btn = tk.Button(self.sidebar, text=f"  {m}", bg=BG_CARD, fg=TEXT_PRIMARY,
                                font=("Segoe UI", 10), anchor="w", bd=0, padx=10, pady=8,
                                activebackground=BG_INPUT, activeforeground=ACCENT_LIGHT,
                                cursor="hand2", relief="flat",
                                command=lambda m=m: self.show_method(m))
                btn.pack(fill="x", padx=8, pady=1)
                self.method_btns[m] = btn

    # ── shared UI helpers (all unchanged) ─────────────────────────────────────

    def _make_input_row(self, parent, label, default="", width=28, tooltip=""):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 9), width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=default)
        ent = tk.Entry(row, textvariable=var, bg=BG_INPUT, fg=TEXT_PRIMARY,
                       insertbackground=ACCENT_LIGHT, relief="flat",
                       font=FONT_MONO, width=width,
                       highlightthickness=1, highlightbackground=BORDER_COLOR,
                       highlightcolor=ACCENT_BLUE)
        ent.pack(side="left", ipady=4, padx=4)
        return var

    def _make_result_box(self, parent):
        frame = tk.Frame(parent, bg=BG_CARD, bd=0)
        frame.pack(fill="x", pady=(8, 4))
        tk.Label(frame, text="RESULTS", bg=BG_CARD, fg=ACCENT_LIGHT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4, pady=(6, 2))
        tk.Frame(frame, bg=BORDER_COLOR, height=1).pack(fill="x")
        text = tk.Text(frame, bg=BG_INPUT, fg=ACCENT_GREEN, font=FONT_MONO,
                       height=4, relief="flat", state="disabled",
                       highlightthickness=1, highlightbackground=BORDER_COLOR,
                       wrap="none")
        text.pack(fill="x", padx=4, pady=6)
        return text

    def _make_table(self, parent, columns, col_widths=None):
        frame = tk.Frame(parent, bg=BG_CARD)
        frame.pack(fill="both", expand=True, pady=(4, 0))
        tk.Label(frame, text="ITERATION TABLE", bg=BG_CARD, fg=ACCENT_LIGHT,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4, pady=(6, 2))
        tk.Frame(frame, bg=BORDER_COLOR, height=1).pack(fill="x")
        table_frame = tk.Frame(frame, bg=BG_TABLE)
        table_frame.pack(fill="both", expand=True, pady=4)
        vsb = ttk.Scrollbar(table_frame, orient="vertical")
        hsb = ttk.Scrollbar(table_frame, orient="horizontal")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings",
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
                            selectmode="browse")
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        for i, col in enumerate(columns):
            w = col_widths[i] if col_widths else 110
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center", minwidth=60)
        return tree

    def _make_buttons(self, parent, run_cmd, reset_cmd):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=(10, 6))
        tk.Button(row, text="▶  Run", bg=ACCENT_BLUE, fg=TEXT_PRIMARY,
                  font=FONT_BTN, relief="flat", padx=18, pady=7,
                  activebackground="#388bfd", cursor="hand2",
                  command=run_cmd).pack(side="left", padx=(4, 6))
        tk.Button(row, text="↺  Reset", bg=BG_INPUT, fg=TEXT_MUTED,
                  font=FONT_BTN, relief="flat", padx=18, pady=7,
                  activebackground="#2d333b", cursor="hand2",
                  command=reset_cmd).pack(side="left")

    def _set_result(self, text_widget, content, color=ACCENT_GREEN):
        text_widget.config(state="normal")
        text_widget.delete("1.0", "end")
        text_widget.insert("end", content)
        text_widget.config(fg=color, state="disabled")

    def _clear_table(self, tree):
        for row in tree.get_children():
            tree.delete(row)

    def _make_plot_panel(self, parent):
        frame = tk.Frame(parent, bg=BG_DARK)
        fig = Figure(figsize=(5.5, 4.2), dpi=88, facecolor=BG_DARK)
        ax  = fig.add_subplot(111)
        ax.set_facecolor(BG_DARK)
        ax.set_title("Graph will appear after running", color=TEXT_MUTED, fontsize=11)
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
        ax.tick_params(colors=TEXT_MUTED)
        ax.grid(True, color=BORDER_COLOR, linewidth=0.5, alpha=0.4)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        return frame, fig, ax, canvas

    # ── BISECTION ─────────────────────────────────────────────────────────────

    def _build_bisection_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Bisection"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)

        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Bisection Method", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Root finding · Bracketing method",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.bis_fx  = self._make_input_row(inp_frame, "f(x) =", "x**3 - x - 2")
        self.bis_a   = self._make_input_row(inp_frame, "a (lower):", "1")
        self.bis_b   = self._make_input_row(inp_frame, "b (upper):", "2")
        self.bis_tol = self._make_input_row(inp_frame, "Tolerance:", "0.001")
        self.bis_max = self._make_input_row(inp_frame, "Max Iterations:", "100")

        self._make_buttons(left, self._run_bisection, self._reset_bisection)
        self.bis_result = self._make_result_box(left)

        top_right = tk.Frame(right, bg=BG_DARK)
        top_right.pack(fill="both", expand=True)
        plot_frame, self.bis_fig, self.bis_ax, self.bis_canvas = self._make_plot_panel(top_right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))

        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        cols = ("Iter", "a", "b", "c (midpoint)", "f(c)", "Error %")
        self.bis_table = self._make_table(table_outer, cols, [50, 100, 100, 120, 120, 100])

    def _run_bisection(self):
        try:
            fx  = self.bis_fx.get().strip()
            a   = float(self.bis_a.get())
            b   = float(self.bis_b.get())
            tol = float(self.bis_tol.get())
            mxi = int(self.bis_max.get())
            root, iters, ferr, rows = bisection_method(fx, a, b, tol, mxi)
            self._set_result(self.bis_result,
                f"  Root       : {root:.8f}\n"
                f"  Iterations : {iters}\n"
                f"  |f(root)|  : {ferr:.2e}")
            self._clear_table(self.bis_table)
            for r in rows:
                it, ra, rb, rc, fc, err = r
                err_str = f"{err:.6f}" if isinstance(err, float) and not np.isnan(err) else str(err)
                self.bis_table.insert("", "end", values=(it, f"{ra:.6f}", f"{rb:.6f}", f"{rc:.6f}", f"{fc:.6e}", err_str))
            plot_root_function(self.bis_ax, fx, root, "Bisection Method", a=a, b=b)
            self.bis_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_bisection(self):
        self.bis_fx.set("x**3 - x - 2")
        self.bis_a.set("1"); self.bis_b.set("2")
        self.bis_tol.set("0.001"); self.bis_max.set("100")
        self._set_result(self.bis_result, "", ACCENT_GREEN)
        self._clear_table(self.bis_table)
        self.bis_ax.clear()
        self.bis_ax.set_facecolor(BG_DARK)
        self.bis_ax.set_title("Graph will appear after running", color=TEXT_MUTED, fontsize=11)
        self.bis_canvas.draw()

    # ── FALSE POSITION ────────────────────────────────────────────────────────

    def _build_false_position_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["False Position"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="False Position Method", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Root finding · Regula Falsi",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.fp_fx  = self._make_input_row(inp_frame, "f(x) =", "x**3 - x - 2")
        self.fp_a   = self._make_input_row(inp_frame, "a (lower):", "1")
        self.fp_b   = self._make_input_row(inp_frame, "b (upper):", "2")
        self.fp_tol = self._make_input_row(inp_frame, "Tolerance:", "0.001")
        self.fp_max = self._make_input_row(inp_frame, "Max Iterations:", "100")
        self._make_buttons(left, self._run_fp, self._reset_fp)
        self.fp_result = self._make_result_box(left)

        plot_frame, self.fp_fig, self.fp_ax, self.fp_canvas = self._make_plot_panel(right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))
        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        cols = ("Iter", "a", "b", "c (false pos)", "f(c)", "Error %")
        self.fp_table = self._make_table(table_outer, cols, [50, 100, 100, 120, 120, 100])

    def _run_fp(self):
        try:
            fx  = self.fp_fx.get().strip()
            a   = float(self.fp_a.get())
            b   = float(self.fp_b.get())
            tol = float(self.fp_tol.get())
            mxi = int(self.fp_max.get())
            root, iters, ferr, rows = false_position_method(fx, a, b, tol, mxi)
            self._set_result(self.fp_result,
                f"  Root       : {root:.8f}\n"
                f"  Iterations : {iters}\n"
                f"  |f(root)|  : {ferr:.2e}")
            self._clear_table(self.fp_table)
            for r in rows:
                it, ra, rb, rc, fc, err = r
                err_str = f"{err:.6f}" if isinstance(err, float) and not np.isnan(err) else str(err)
                self.fp_table.insert("", "end", values=(it, f"{ra:.6f}", f"{rb:.6f}", f"{rc:.6f}", f"{fc:.6e}", err_str))
            plot_root_function(self.fp_ax, fx, root, "False Position Method", a=a, b=b)
            self.fp_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_fp(self):
        self.fp_fx.set("x**3 - x - 2")
        self.fp_a.set("1"); self.fp_b.set("2")
        self.fp_tol.set("0.001"); self.fp_max.set("100")
        self._set_result(self.fp_result, "", ACCENT_GREEN)
        self._clear_table(self.fp_table)
        self.fp_ax.clear(); self.fp_canvas.draw()

    # ── NEWTON-RAPHSON ────────────────────────────────────────────────────────

    def _build_newton_raphson_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Newton-Raphson"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Newton-Raphson Method", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Root finding · Open method (uses derivative)",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.nr_fx  = self._make_input_row(inp_frame, "f(x) =", "x**3 - x - 2")
        self.nr_x0  = self._make_input_row(inp_frame, "x₀ (initial):", "1.5")
        self.nr_tol = self._make_input_row(inp_frame, "Tolerance:", "0.001")
        self.nr_max = self._make_input_row(inp_frame, "Max Iterations:", "50")

        self.nr_df_label = tk.Label(left, text="f'(x): (computed automatically)",
                                     bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL)
        self.nr_df_label.pack(anchor="w", padx=14, pady=(0, 4))

        self._make_buttons(left, self._run_nr, self._reset_nr)
        self.nr_result = self._make_result_box(left)

        plot_frame, self.nr_fig, self.nr_ax, self.nr_canvas = self._make_plot_panel(right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))
        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        cols = ("Iter", "xᵢ", "f(xᵢ)", "f'(xᵢ)", "xᵢ₊₁", "Error %")
        self.nr_table = self._make_table(table_outer, cols, [50, 110, 110, 110, 110, 90])

    def _run_nr(self):
        try:
            fx  = self.nr_fx.get().strip()
            x0  = float(self.nr_x0.get())
            tol = float(self.nr_tol.get())
            mxi = int(self.nr_max.get())
            expr, x_sym = get_sympy_expr(fx)
            df_str = str(sp.diff(expr, x_sym))
            self.nr_df_label.config(text=f"  f'(x) = {df_str}", fg=ACCENT_YELLOW)
            root, iters, ferr, rows = newton_raphson_method(fx, x0, tol, mxi)
            self._set_result(self.nr_result,
                f"  Root       : {root:.8f}\n"
                f"  Iterations : {iters}\n"
                f"  |f(root)|  : {ferr:.2e}")
            self._clear_table(self.nr_table)
            for r in rows:
                it, xi, fxi, dfxi, xi_new, err = r
                self.nr_table.insert("", "end", values=(
                    it, f"{xi:.6f}", f"{fxi:.6e}", f"{dfxi:.6e}", f"{xi_new:.6f}", f"{err:.6f}"))
            plot_root_function(self.nr_ax, fx, root, "Newton-Raphson Method", a=x0-2, b=x0+2)
            self.nr_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_nr(self):
        self.nr_fx.set("x**3 - x - 2")
        self.nr_x0.set("1.5")
        self.nr_tol.set("0.001"); self.nr_max.set("50")
        self.nr_df_label.config(text="f'(x): (computed automatically)", fg=TEXT_MUTED)
        self._set_result(self.nr_result, "", ACCENT_GREEN)
        self._clear_table(self.nr_table)
        self.nr_ax.clear(); self.nr_canvas.draw()

    # ── SECANT ────────────────────────────────────────────────────────────────

    def _build_secant_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Secant"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Secant Method", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Root finding · Open method (no derivative needed)",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.sc_fx  = self._make_input_row(inp_frame, "f(x) =", "x**3 - x - 2")
        self.sc_x0  = self._make_input_row(inp_frame, "x₀:", "1")
        self.sc_x1  = self._make_input_row(inp_frame, "x₁:", "2")
        self.sc_tol = self._make_input_row(inp_frame, "Tolerance:", "0.001")
        self.sc_max = self._make_input_row(inp_frame, "Max Iterations:", "50")
        self._make_buttons(left, self._run_secant, self._reset_secant)
        self.sc_result = self._make_result_box(left)

        plot_frame, self.sc_fig, self.sc_ax, self.sc_canvas = self._make_plot_panel(right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))
        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        cols = ("Iter", "xᵢ₋₁", "xᵢ", "f(xᵢ)", "xᵢ₊₁", "Error %")
        self.sc_table = self._make_table(table_outer, cols, [50, 110, 110, 110, 110, 90])

    def _run_secant(self):
        try:
            fx  = self.sc_fx.get().strip()
            x0  = float(self.sc_x0.get())
            x1  = float(self.sc_x1.get())
            tol = float(self.sc_tol.get())
            mxi = int(self.sc_max.get())
            root, iters, ferr, rows = secant_method(fx, x0, x1, tol, mxi)
            self._set_result(self.sc_result,
                f"  Root       : {root:.8f}\n"
                f"  Iterations : {iters}\n"
                f"  |f(root)|  : {ferr:.2e}")
            self._clear_table(self.sc_table)
            for r in rows:
                it, xi1, xi, fxi, xi_new, err = r
                self.sc_table.insert("", "end", values=(
                    it, f"{xi1:.6f}", f"{xi:.6f}", f"{fxi:.6e}", f"{xi_new:.6f}", f"{err:.6f}"))
            plot_root_function(self.sc_ax, fx, root, "Secant Method", a=x0, b=x1)
            self.sc_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_secant(self):
        self.sc_fx.set("x**3 - x - 2")
        self.sc_x0.set("1"); self.sc_x1.set("2")
        self.sc_tol.set("0.001"); self.sc_max.set("50")
        self._set_result(self.sc_result, "", ACCENT_GREEN)
        self._clear_table(self.sc_table)
        self.sc_ax.clear(); self.sc_canvas.draw()

    # ── GAUSS-SEIDEL ──────────────────────────────────────────────────────────

    def _build_gauss_seidel_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Gauss-Seidel"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=400)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Gauss-Seidel Method", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Linear systems · Iterative solver",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 6))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)
        self.gs_matrix_frame, self.gs_n_var = self._build_matrix_input(left, "gs")
        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=4)
        self.gs_tol = self._make_input_row(inp_frame, "Tolerance:", "0.0001")
        self.gs_max = self._make_input_row(inp_frame, "Max Iterations:", "50")
        self._make_buttons(left, self._run_gs, self._reset_gs)
        self.gs_result = self._make_result_box(left)

        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="both", expand=True)
        self.gs_iter_label = tk.Label(table_outer, text="ITERATION TABLE",
                                       bg=BG_CARD, fg=ACCENT_LIGHT, font=("Segoe UI", 8, "bold"))
        self.gs_iter_label.pack(anchor="w", padx=4, pady=(8, 2))
        tk.Frame(table_outer, bg=BORDER_COLOR, height=1).pack(fill="x")
        self.gs_table_frame = tk.Frame(table_outer, bg=BG_TABLE)
        self.gs_table_frame.pack(fill="both", expand=True, pady=4)
        self.gs_table = None

    def _build_matrix_input(self, parent, prefix):
        frame = tk.Frame(parent, bg=BG_CARD)
        frame.pack(fill="x", padx=14, pady=6)
        top = tk.Frame(frame, bg=BG_CARD)
        top.pack(fill="x")
        tk.Label(top, text="Matrix size (n×n):", bg=BG_CARD,
                 fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(side="left")
        n_var = tk.StringVar(value="3")
        n_entry = tk.Entry(top, textvariable=n_var, bg=BG_INPUT, fg=TEXT_PRIMARY,
                           font=FONT_MONO, width=4, relief="flat",
                           highlightthickness=1, highlightbackground=BORDER_COLOR,
                           highlightcolor=ACCENT_BLUE)
        n_entry.pack(side="left", padx=6, ipady=3)
        tk.Button(top, text="Set Size", bg=BG_INPUT, fg=ACCENT_LIGHT,
                  font=("Segoe UI", 9), relief="flat", padx=8, pady=3,
                  cursor="hand2",
                  command=lambda: self._rebuild_matrix(frame, n_var, prefix)).pack(side="left", padx=4)
        entries_frame = tk.Frame(frame, bg=BG_CARD)
        entries_frame.pack(fill="x", pady=4)
        setattr(self, f"{prefix}_entries_frame", entries_frame)
        self._rebuild_matrix(frame, n_var, prefix)
        return frame, n_var

    def _rebuild_matrix(self, parent, n_var, prefix):
        try:
            n = int(n_var.get())
            assert 2 <= n <= 6
        except:
            messagebox.showerror("Error", "Matrix size must be between 2 and 6.")
            return
        entries_frame = getattr(self, f"{prefix}_entries_frame")
        for w in entries_frame.winfo_children():
            w.destroy()
        tk.Label(entries_frame, text="Matrix A (coefficients):", bg=BG_CARD,
                 fg=TEXT_MUTED, font=FONT_SMALL).grid(row=0, column=0, columnspan=n+1, sticky="w", pady=(4, 2))
        a_entries = []
        for i in range(n):
            row_entries = []
            for j in range(n):
                default = "4" if i == j else "1"
                e = tk.Entry(entries_frame, bg=BG_INPUT, fg=TEXT_PRIMARY, font=FONT_MONO,
                             width=6, relief="flat", justify="center",
                             highlightthickness=1, highlightbackground=BORDER_COLOR)
                e.insert(0, default)
                e.grid(row=i+1, column=j, padx=2, pady=2, ipady=3)
                row_entries.append(e)
            a_entries.append(row_entries)
        tk.Label(entries_frame, text=" b", bg=BG_CARD,
                 fg=ACCENT_YELLOW, font=("Segoe UI", 9, "bold")).grid(row=0, column=n+1, padx=(8, 0))
        b_entries = []
        for i in range(n):
            default_b = str((i + 1) * 5)
            e = tk.Entry(entries_frame, bg=BG_INPUT, fg=ACCENT_YELLOW, font=FONT_MONO,
                         width=6, relief="flat", justify="center",
                         highlightthickness=1, highlightbackground=BORDER_COLOR)
            e.insert(0, default_b)
            e.grid(row=i+1, column=n+1, padx=(8, 2), pady=2, ipady=3)
            b_entries.append(e)
        setattr(self, f"{prefix}_a_entries", a_entries)
        setattr(self, f"{prefix}_b_entries", b_entries)

    def _get_matrix(self, prefix):
        a_entries = getattr(self, f"{prefix}_a_entries")
        b_entries = getattr(self, f"{prefix}_b_entries")
        n = len(a_entries)
        A = [[float(a_entries[i][j].get()) for j in range(n)] for i in range(n)]
        b = [float(b_entries[i].get()) for i in range(n)]
        return np.array(A, dtype=float), np.array(b, dtype=float)

    def _display_iter_table(self, table_frame, rows, n, old_tree_attr):
        old = getattr(self, old_tree_attr, None)
        if old:
            try:
                old.destroy()
            except:
                pass
        for w in table_frame.winfo_children():
            w.destroy()
        cols = ["Iter"] + [f"x{i+1}" for i in range(n)] + ["‖error‖∞"]
        vsb = ttk.Scrollbar(table_frame, orient="vertical")
        hsb = ttk.Scrollbar(table_frame, orient="horizontal")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=90, anchor="center")
        for r in rows:
            it, xvec, err = r
            vals = [it] + [f"{v:.6f}" for v in xvec] + [f"{err:.6e}"]
            tree.insert("", "end", values=vals)
        setattr(self, old_tree_attr, tree)

    def _run_gs(self):
        try:
            A, b = self._get_matrix("gs")
            tol = float(self.gs_tol.get())
            mxi = int(self.gs_max.get())
            x, iters, rows = gauss_seidel_method(A.tolist(), b.tolist(), tol, mxi)
            sol_str = "  Solution Vector:\n" + "\n".join(f"  x{i+1} = {v:.8f}" for i, v in enumerate(x))
            sol_str += f"\n  Iterations : {iters}"
            self._set_result(self.gs_result, sol_str)
            self._display_iter_table(self.gs_table_frame, rows, len(x), "gs_tree")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_gs(self):
        self._rebuild_matrix(None, self.gs_n_var, "gs")
        self.gs_tol.set("0.0001"); self.gs_max.set("50")
        self._set_result(self.gs_result, "", ACCENT_GREEN)

    # ── JACOBI ────────────────────────────────────────────────────────────────

    def _build_jacobi_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Jacobi"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=400)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Jacobi Method", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Linear systems · Iterative solver (simultaneous update)",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 6))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)
        self.jac_matrix_frame, self.jac_n_var = self._build_matrix_input(left, "jac")
        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=4)
        self.jac_tol = self._make_input_row(inp_frame, "Tolerance:", "0.0001")
        self.jac_max = self._make_input_row(inp_frame, "Max Iterations:", "50")
        self._make_buttons(left, self._run_jacobi, self._reset_jacobi)
        self.jac_result = self._make_result_box(left)

        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="both", expand=True)
        tk.Label(table_outer, text="ITERATION TABLE",
                 bg=BG_CARD, fg=ACCENT_LIGHT, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4, pady=(8, 2))
        tk.Frame(table_outer, bg=BORDER_COLOR, height=1).pack(fill="x")
        self.jac_table_frame = tk.Frame(table_outer, bg=BG_TABLE)
        self.jac_table_frame.pack(fill="both", expand=True, pady=4)
        self.jac_tree = None

    def _run_jacobi(self):
        try:
            A, b = self._get_matrix("jac")
            tol = float(self.jac_tol.get())
            mxi = int(self.jac_max.get())
            x, iters, rows = jacobi_method(A.tolist(), b.tolist(), tol, mxi)
            sol_str = "  Solution Vector:\n" + "\n".join(f"  x{i+1} = {v:.8f}" for i, v in enumerate(x))
            sol_str += f"\n  Iterations : {iters}"
            self._set_result(self.jac_result, sol_str)
            self._display_iter_table(self.jac_table_frame, rows, len(x), "jac_tree")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_jacobi(self):
        self._rebuild_matrix(None, self.jac_n_var, "jac")
        self.jac_tol.set("0.0001"); self.jac_max.set("50")
        self._set_result(self.jac_result, "", ACCENT_GREEN)

    # ═══════════════════════════════════════════════════════════════════════════
    #  NEW ── Shared interpolation helper: finite difference table display
    # ═══════════════════════════════════════════════════════════════════════════

    def _show_diff_table(self, container, diff, x_vals, tree_attr):
        """Render a finite difference table inside container (clears first)."""
        for w in container.winfo_children():
            w.destroy()
        n = len(x_vals)
        cols = ["x", "Δ⁰y"] + [f"Δ{k}y" for k in range(1, n)]
        vsb = ttk.Scrollbar(container, orient="vertical")
        hsb = ttk.Scrollbar(container, orient="horizontal")
        tree = ttk.Treeview(container, columns=cols, show="headings",
                            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(fill="both", expand=True)
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=88, anchor="center", minwidth=60)
        for i in range(n):
            row_vals = [f"{x_vals[i]:.4f}", f"{diff[i][0]:.6f}"]
            for j in range(1, n):
                if i + j <= n - 1:
                    row_vals.append(f"{diff[i][j]:.6f}")
                else:
                    row_vals.append("")
            tree.insert("", "end", values=row_vals)
        setattr(self, tree_attr, tree)

    # ═══════════════════════════════════════════════════════════════════════════
    #  NEW ── Newton Forward Frame
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_newton_forward_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Newton Forward"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Newton Forward Interpolation", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Interpolation · Equal spacing · Forward differences",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.nf_x   = self._make_input_row(inp_frame, "x values:", "0, 1, 2, 3, 4", width=30)
        self.nf_y   = self._make_input_row(inp_frame, "y values:", "1, 2, 4, 8, 16", width=30)
        self.nf_xt  = self._make_input_row(inp_frame, "Target x:", "2.5")

        # s value display (computed)
        self.nf_s_label = tk.Label(left, text="s: (computed after run)",
                                    bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL)
        self.nf_s_label.pack(anchor="w", padx=14, pady=(0, 4))

        self._make_buttons(left, self._run_newton_forward, self._reset_newton_forward)
        self.nf_result = self._make_result_box(left)

        # Right: plot on top, difference table below
        plot_frame, self.nf_fig, self.nf_ax, self.nf_canvas = self._make_plot_panel(right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))

        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        tk.Label(table_outer, text="FORWARD DIFFERENCE TABLE", bg=BG_CARD,
                 fg=ACCENT_LIGHT, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4, pady=(6, 2))
        tk.Frame(table_outer, bg=BORDER_COLOR, height=1).pack(fill="x")
        self.nf_diff_frame = tk.Frame(table_outer, bg=BG_TABLE, height=130)
        self.nf_diff_frame.pack(fill="x", pady=4)
        self.nf_diff_frame.pack_propagate(False)
        self.nf_diff_tree = None

        # Terms contribution table
        terms_outer = tk.Frame(right, bg=BG_CARD)
        terms_outer.pack(fill="x", pady=(4, 0))
        cols = ("Order k", "Δᵏy₀", "Coefficient", "Contribution")
        self.nf_terms_table = self._make_table(terms_outer, cols, [80, 130, 130, 130])

    def _run_newton_forward(self):
        try:
            x_vals = np.array(_parse_float_list(self.nf_x.get()))
            y_vals = np.array(_parse_float_list(self.nf_y.get()))
            if len(x_vals) != len(y_vals):
                raise ValueError("x and y must have the same number of values.")
            if len(x_vals) < 2:
                raise ValueError("At least 2 data points are required.")
            x_target = float(self.nf_xt.get())

            result, diff, s, terms = newton_forward_interp(x_vals.tolist(), y_vals.tolist(), x_target)

            self.nf_s_label.config(text=f"  s = (x - x₀)/h = {s:.6f}", fg=ACCENT_YELLOW)
            self._set_result(self.nf_result,
                f"  Interpolated f({x_target}) : {result:.8f}\n"
                f"  s value                   : {s:.6f}\n"
                f"  Data points used          : {len(x_vals)}")

            # Difference table
            self._show_diff_table(self.nf_diff_frame, diff, x_vals.tolist(), "nf_diff_tree")

            # Terms table
            self._clear_table(self.nf_terms_table)
            for (k, delta, coeff, contrib) in terms:
                self.nf_terms_table.insert("", "end", values=(
                    k, f"{delta:.6f}", f"{coeff:.6f}", f"{contrib:.6f}"))

            # Plot
            def interp_callable(xp):
                r, _, _, _ = newton_forward_interp(x_vals.tolist(), y_vals.tolist(), xp)
                return r
            plot_interpolation(self.nf_ax, x_vals.tolist(), y_vals.tolist(),
                               x_target, result, "Newton Forward Interpolation", interp_callable)
            self.nf_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_newton_forward(self):
        self.nf_x.set("0, 1, 2, 3, 4")
        self.nf_y.set("1, 2, 4, 8, 16")
        self.nf_xt.set("2.5")
        self.nf_s_label.config(text="s: (computed after run)", fg=TEXT_MUTED)
        self._set_result(self.nf_result, "", ACCENT_GREEN)
        for w in self.nf_diff_frame.winfo_children():
            w.destroy()
        self._clear_table(self.nf_terms_table)
        self.nf_ax.clear()
        self.nf_ax.set_facecolor(BG_DARK)
        self.nf_ax.set_title("Graph will appear after running", color=TEXT_MUTED, fontsize=11)
        self.nf_canvas.draw()

    # ═══════════════════════════════════════════════════════════════════════════
    #  NEW ── Newton Backward Frame
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_newton_backward_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Newton Backward"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Newton Backward Interpolation", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Interpolation · Equal spacing · Backward differences",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.nb_x   = self._make_input_row(inp_frame, "x values:", "0, 1, 2, 3, 4", width=30)
        self.nb_y   = self._make_input_row(inp_frame, "y values:", "1, 2, 4, 8, 16", width=30)
        self.nb_xt  = self._make_input_row(inp_frame, "Target x:", "3.5")

        self.nb_s_label = tk.Label(left, text="s: (computed after run)",
                                    bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL)
        self.nb_s_label.pack(anchor="w", padx=14, pady=(0, 4))

        self._make_buttons(left, self._run_newton_backward, self._reset_newton_backward)
        self.nb_result = self._make_result_box(left)

        plot_frame, self.nb_fig, self.nb_ax, self.nb_canvas = self._make_plot_panel(right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))

        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        tk.Label(table_outer, text="BACKWARD DIFFERENCE TABLE", bg=BG_CARD,
                 fg=ACCENT_LIGHT, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4, pady=(6, 2))
        tk.Frame(table_outer, bg=BORDER_COLOR, height=1).pack(fill="x")
        self.nb_diff_frame = tk.Frame(table_outer, bg=BG_TABLE, height=130)
        self.nb_diff_frame.pack(fill="x", pady=4)
        self.nb_diff_frame.pack_propagate(False)
        self.nb_diff_tree = None

        terms_outer = tk.Frame(right, bg=BG_CARD)
        terms_outer.pack(fill="x", pady=(4, 0))
        cols = ("Order k", "∇ᵏyₙ", "Coefficient", "Contribution")
        self.nb_terms_table = self._make_table(terms_outer, cols, [80, 130, 130, 130])

    def _run_newton_backward(self):
        try:
            x_vals = np.array(_parse_float_list(self.nb_x.get()))
            y_vals = np.array(_parse_float_list(self.nb_y.get()))
            if len(x_vals) != len(y_vals):
                raise ValueError("x and y must have the same number of values.")
            if len(x_vals) < 2:
                raise ValueError("At least 2 data points are required.")
            x_target = float(self.nb_xt.get())

            result, diff, s, terms = newton_backward_interp(x_vals.tolist(), y_vals.tolist(), x_target)

            self.nb_s_label.config(text=f"  s = (x - xₙ)/h = {s:.6f}", fg=ACCENT_YELLOW)
            self._set_result(self.nb_result,
                f"  Interpolated f({x_target}) : {result:.8f}\n"
                f"  s value                   : {s:.6f}\n"
                f"  Data points used          : {len(x_vals)}")

            self._show_diff_table(self.nb_diff_frame, diff, x_vals.tolist(), "nb_diff_tree")

            self._clear_table(self.nb_terms_table)
            for (k, delta, coeff, contrib) in terms:
                self.nb_terms_table.insert("", "end", values=(
                    k, f"{delta:.6f}", f"{coeff:.6f}", f"{contrib:.6f}"))

            def interp_callable(xp):
                r, _, _, _ = newton_backward_interp(x_vals.tolist(), y_vals.tolist(), xp)
                return r
            plot_interpolation(self.nb_ax, x_vals.tolist(), y_vals.tolist(),
                               x_target, result, "Newton Backward Interpolation", interp_callable)
            self.nb_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_newton_backward(self):
        self.nb_x.set("0, 1, 2, 3, 4")
        self.nb_y.set("1, 2, 4, 8, 16")
        self.nb_xt.set("3.5")
        self.nb_s_label.config(text="s: (computed after run)", fg=TEXT_MUTED)
        self._set_result(self.nb_result, "", ACCENT_GREEN)
        for w in self.nb_diff_frame.winfo_children():
            w.destroy()
        self._clear_table(self.nb_terms_table)
        self.nb_ax.clear()
        self.nb_ax.set_facecolor(BG_DARK)
        self.nb_ax.set_title("Graph will appear after running", color=TEXT_MUTED, fontsize=11)
        self.nb_canvas.draw()

    # ═══════════════════════════════════════════════════════════════════════════
    #  NEW ── Stirling's Interpolation Frame
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_stirling_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Stirling"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Stirling's Interpolation", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Interpolation · Equal spacing · Central differences",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.st_x  = self._make_input_row(inp_frame, "x values:", "10, 20, 30, 40, 50", width=30)
        self.st_y  = self._make_input_row(inp_frame, "y values:", "1.1, 2.3, 3.6, 5.0, 6.7", width=30)
        self.st_xt = self._make_input_row(inp_frame, "Target x:", "25")

        self.st_info_label = tk.Label(left, text="Central index: (computed after run)",
                                       bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL)
        self.st_info_label.pack(anchor="w", padx=14, pady=(0, 4))

        self._make_buttons(left, self._run_stirling, self._reset_stirling)
        self.st_result = self._make_result_box(left)

        plot_frame, self.st_fig, self.st_ax, self.st_canvas = self._make_plot_panel(right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))

        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        tk.Label(table_outer, text="CENTRAL DIFFERENCE TABLE", bg=BG_CARD,
                 fg=ACCENT_LIGHT, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=4, pady=(6, 2))
        tk.Frame(table_outer, bg=BORDER_COLOR, height=1).pack(fill="x")
        self.st_diff_frame = tk.Frame(table_outer, bg=BG_TABLE, height=130)
        self.st_diff_frame.pack(fill="x", pady=4)
        self.st_diff_frame.pack_propagate(False)
        self.st_diff_tree = None

        terms_outer = tk.Frame(right, bg=BG_CARD)
        terms_outer.pack(fill="x", pady=(4, 0))
        cols = ("Order", "Diff. Value", "Term Label", "Coefficient", "Contribution")
        self.st_terms_table = self._make_table(terms_outer, cols, [65, 110, 110, 110, 110])

    def _run_stirling(self):
        try:
            x_vals = np.array(_parse_float_list(self.st_x.get()))
            y_vals = np.array(_parse_float_list(self.st_y.get()))
            if len(x_vals) != len(y_vals):
                raise ValueError("x and y must have the same number of values.")
            if len(x_vals) < 3:
                raise ValueError("Stirling's formula requires at least 3 data points.")
            x_target = float(self.st_xt.get())

            result, diff, mid, s, terms = stirling_interp(x_vals.tolist(), y_vals.tolist(), x_target)

            self.st_info_label.config(
                text=f"  Central x₀ = {x_vals[mid]:.4g}  |  s = {s:.6f}", fg=ACCENT_YELLOW)
            self._set_result(self.st_result,
                f"  Interpolated f({x_target}) : {result:.8f}\n"
                f"  Central index x₀          : {x_vals[mid]:.4g}\n"
                f"  s value                   : {s:.6f}")

            self._show_diff_table(self.st_diff_frame, diff, x_vals.tolist(), "st_diff_tree")

            self._clear_table(self.st_terms_table)
            for (order, diff_val, label, coeff, contrib) in terms:
                self.st_terms_table.insert("", "end", values=(
                    order, f"{diff_val:.6f}", label, f"{coeff:.6f}", f"{contrib:.6f}"))

            def interp_callable(xp):
                r, _, _, _, _ = stirling_interp(x_vals.tolist(), y_vals.tolist(), xp)
                return r
            plot_interpolation(self.st_ax, x_vals.tolist(), y_vals.tolist(),
                               x_target, result, "Stirling's Interpolation", interp_callable)
            self.st_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_stirling(self):
        self.st_x.set("10, 20, 30, 40, 50")
        self.st_y.set("1.1, 2.3, 3.6, 5.0, 6.7")
        self.st_xt.set("25")
        self.st_info_label.config(text="Central index: (computed after run)", fg=TEXT_MUTED)
        self._set_result(self.st_result, "", ACCENT_GREEN)
        for w in self.st_diff_frame.winfo_children():
            w.destroy()
        self._clear_table(self.st_terms_table)
        self.st_ax.clear()
        self.st_ax.set_facecolor(BG_DARK)
        self.st_ax.set_title("Graph will appear after running", color=TEXT_MUTED, fontsize=11)
        self.st_canvas.draw()

    # ═══════════════════════════════════════════════════════════════════════════
    #  NEW ── Lagrange Interpolation Frame
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_lagrange_frame(self):
        outer = tk.Frame(self.content, bg=BG_DARK)
        self.method_frames["Lagrange"] = outer

        left = tk.Frame(outer, bg=BG_CARD, width=370)
        left.pack(side="left", fill="y", padx=(10, 4), pady=10)
        left.pack_propagate(False)
        right = tk.Frame(outer, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True, padx=(4, 10), pady=10)

        tk.Label(left, text="Lagrange Interpolation", bg=BG_CARD,
                 fg=TEXT_PRIMARY, font=FONT_TITLE).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(left, text="Interpolation · Unequal spacing supported",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 10))
        tk.Frame(left, bg=BORDER_COLOR, height=1).pack(fill="x", padx=10)

        inp_frame = tk.Frame(left, bg=BG_CARD)
        inp_frame.pack(fill="x", padx=14, pady=8)
        self.lg_x  = self._make_input_row(inp_frame, "x values:", "0, 1, 3, 4", width=30)
        self.lg_y  = self._make_input_row(inp_frame, "y values:", "1, 3, 2, 5", width=30)
        self.lg_xt = self._make_input_row(inp_frame, "Target x:", "2")

        tk.Label(left, text="  Spacing: any (equal or unequal)",
                 bg=BG_CARD, fg=TEXT_MUTED, font=FONT_SMALL).pack(anchor="w", padx=14, pady=(0, 4))

        self._make_buttons(left, self._run_lagrange, self._reset_lagrange)
        self.lg_result = self._make_result_box(left)

        plot_frame, self.lg_fig, self.lg_ax, self.lg_canvas = self._make_plot_panel(right)
        plot_frame.pack(fill="both", expand=True, pady=(0, 4))

        table_outer = tk.Frame(right, bg=BG_CARD)
        table_outer.pack(fill="x", pady=(4, 0))
        cols = ("k", "xₖ", "yₖ", "Lₖ(x*)", "Contribution")
        self.lg_table = self._make_table(table_outer, cols, [50, 100, 100, 130, 130])

    def _run_lagrange(self):
        try:
            x_vals = np.array(_parse_float_list(self.lg_x.get()))
            y_vals = np.array(_parse_float_list(self.lg_y.get()))
            if len(x_vals) != len(y_vals):
                raise ValueError("x and y must have the same number of values.")
            if len(x_vals) < 2:
                raise ValueError("At least 2 data points are required.")
            x_target = float(self.lg_xt.get())

            result, terms = lagrange_interp(x_vals.tolist(), y_vals.tolist(), x_target)

            self._set_result(self.lg_result,
                f"  Interpolated f({x_target}) : {result:.8f}\n"
                f"  Polynomial degree         : n - 1 = {len(x_vals) - 1}\n"
                f"  Data points used          : {len(x_vals)}")

            self._clear_table(self.lg_table)
            for (k, xk, yk, Lk, contrib) in terms:
                self.lg_table.insert("", "end", values=(
                    k, f"{xk:.4f}", f"{yk:.4f}", f"{Lk:.6f}", f"{contrib:.6f}"))

            def interp_callable(xp):
                r, _ = lagrange_interp(x_vals.tolist(), y_vals.tolist(), xp)
                return r
            plot_interpolation(self.lg_ax, x_vals.tolist(), y_vals.tolist(),
                               x_target, result, "Lagrange Interpolation", interp_callable)
            self.lg_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _reset_lagrange(self):
        self.lg_x.set("0, 1, 3, 4")
        self.lg_y.set("1, 3, 2, 5")
        self.lg_xt.set("2")
        self._set_result(self.lg_result, "", ACCENT_GREEN)
        self._clear_table(self.lg_table)
        self.lg_ax.clear()
        self.lg_ax.set_facecolor(BG_DARK)
        self.lg_ax.set_title("Graph will appear after running", color=TEXT_MUTED, fontsize=11)
        self.lg_canvas.draw()

    # ── Method switcher (unchanged) ───────────────────────────────────────────

    def show_method(self, method_name):
        if self.current_method:
            self.method_frames[self.current_method].pack_forget()
            btn = self.method_btns.get(self.current_method)
            if btn:
                btn.config(bg=BG_CARD, fg=TEXT_PRIMARY)
        frame = self.method_frames.get(method_name)
        if frame:
            frame.pack(fill="both", expand=True)
        btn = self.method_btns.get(method_name)
        if btn:
            btn.config(bg=BG_INPUT, fg=ACCENT_LIGHT)
        self.current_method = method_name


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(True, True)
    try:
        root.state("zoomed")   # Windows maximize
    except:
        try:
            root.attributes("-zoomed", True)
        except:
            root.geometry("1280x800")
    app = NumericalCalculatorApp(root)
    root.mainloop()
