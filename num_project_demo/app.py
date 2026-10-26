import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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




import streamlit as st
import pandas as pd
from matplotlib.figure import Figure

st.set_page_config(page_title="Advanced Numerical Methods Calculator", page_icon="∑", layout="wide")
st.markdown("""
<style>
.stApp { background: #0d1117; color: #e6edf3; }
[data-testid="stSidebar"] { background: #161b22; }
[data-testid="stMetric"] { background: #161b22; border: 1px solid #30363d; padding: 14px; border-radius: 12px; }
.stButton > button { border-radius: 9px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)
st.title("∑ Advanced Numerical Methods Calculator")
st.caption("Interactive web demo · Root Finding · Linear Systems · Interpolation")

GROUPS = {
    "Root Finding": ["Bisection", "False Position", "Newton-Raphson", "Secant"],
    "Linear Systems": ["Gauss-Seidel", "Jacobi"],
    "Interpolation": ["Newton Forward", "Newton Backward", "Stirling", "Lagrange"],
}
with st.sidebar:
    st.header("Methods")
    group = st.selectbox("Category", list(GROUPS))
    method = st.selectbox("Choose a method", GROUPS[group])
    st.divider()
    st.caption("Tip: use x as the variable, e.g. x**3 - x - 2, sin(x), or exp(x).")


def show_table(rows, columns):
    if rows:
        st.subheader("Iteration / calculation table")
        st.dataframe(pd.DataFrame(rows, columns=columns), use_container_width=True, hide_index=True)


def show_root_plot(expr, root, method_name, a=None, b=None):
    fig = Figure(figsize=(8, 4.2), facecolor="#0d1117")
    ax = fig.subplots()
    plot_root_function(ax, expr, root, method_name, a=a, b=b)
    st.pyplot(fig, use_container_width=True)

if group == "Root Finding":
    st.subheader(f"{method} Method")
    expr = st.text_input("Function f(x)", value="x**3 - x - 2", help="Examples: x**2 - 4, sin(x), exp(x) - 3")
    c1, c2, c3 = st.columns(3)
    tol = c1.number_input("Tolerance (%)", min_value=0.000001, value=0.0001, format="%.6f")
    max_iter = c2.number_input("Maximum iterations", min_value=1, max_value=1000, value=50, step=1)
    if method in ("Bisection", "False Position"):
        a = c1.number_input("Lower bound a", value=1.0, key="root_a")
        b = c2.number_input("Upper bound b", value=2.0, key="root_b")
    elif method == "Newton-Raphson":
        x0 = c1.number_input("Initial guess x₀", value=1.5, key="newton_x0")
    else:
        x0 = c1.number_input("Initial guess x₀", value=1.0, key="secant_x0")
        x1 = c2.number_input("Second guess x₁", value=2.0, key="secant_x1")
    if st.button("Calculate root", type="primary"):
        try:
            if method == "Bisection":
                root, count, residual, rows = bisection_method(expr, float(a), float(b), float(tol), int(max_iter))
                cols = ["Iteration", "a", "b", "c", "f(c)", "Approx. error (%)"]
                plot_a, plot_b = float(a), float(b)
            elif method == "False Position":
                root, count, residual, rows = false_position_method(expr, float(a), float(b), float(tol), int(max_iter))
                cols = ["Iteration", "a", "b", "c", "f(c)", "Approx. error (%)"]
                plot_a, plot_b = float(a), float(b)
            elif method == "Newton-Raphson":
                root, count, residual, rows = newton_raphson_method(expr, float(x0), float(tol), int(max_iter))
                cols = ["Iteration", "xᵢ", "f(xᵢ)", "f′(xᵢ)", "xᵢ₊₁", "Approx. error (%)"]
                plot_a = plot_b = None
            else:
                root, count, residual, rows = secant_method(expr, float(x0), float(x1), float(tol), int(max_iter))
                cols = ["Iteration", "xᵢ₋₁", "xᵢ", "f(xᵢ)", "xᵢ₊₁", "Approx. error (%)"]
                plot_a = plot_b = None
            m1, m2, m3 = st.columns(3)
            m1.metric("Approximate root", f"{root:.10g}")
            m2.metric("Iterations", str(count))
            m3.metric("Residual |f(root)|", f"{residual:.3e}")
            show_table(rows, cols)
            show_root_plot(expr, root, method, plot_a, plot_b)
        except Exception as e:
            st.error(f"Could not calculate: {e}")

elif group == "Linear Systems":
    st.subheader(f"{method} Iterative Solver")
    st.write("Enter a square coefficient matrix A and right-hand-side vector b. Separate values with commas and rows with new lines.")
    default_A = "4, 1, 1\n2, 7, 1\n1, -3, 12"
    default_b = "7,  -8, 6".replace("  ", " ")
    c1, c2 = st.columns(2)
    matrix_text = c1.text_area("Matrix A", value=default_A, height=130)
    vector_text = c2.text_area("Vector b", value=default_b, height=130)
    c3, c4 = st.columns(2)
    tol = c3.number_input("Tolerance", min_value=0.0000001, value=0.0001, format="%.7f")
    max_iter = c4.number_input("Maximum iterations", min_value=1, max_value=10000, value=100, step=1)
    if st.button("Solve system", type="primary"):
        try:
            A = np.array([[float(v.strip()) for v in row.split(",")] for row in matrix_text.strip().splitlines() if row.strip()], dtype=float)
            b = np.array([float(v.strip()) for v in vector_text.replace("\n", ",").split(",") if v.strip()], dtype=float)
            if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != len(b):
                raise ValueError("A must be square, and b must have one value per row of A.")
            if method == "Gauss-Seidel":
                solution, count, rows = gauss_seidel_method(A.tolist(), b.tolist(), float(tol), int(max_iter))
            else:
                solution, count, rows = jacobi_method(A.tolist(), b.tolist(), float(tol), int(max_iter))
            st.success(f"Solution after {count} iterations")
            st.dataframe(pd.DataFrame({"Variable": [f"x{i+1}" for i in range(len(solution))], "Value": solution}), use_container_width=True, hide_index=True)
            iter_rows = [{"Iteration": it, **{f"x{i+1}": val for i, val in enumerate(vec)}, "Error (∞-norm)": err} for it, vec, err in rows]
            show_table(iter_rows, list(iter_rows[0].keys()) if iter_rows else [])
        except Exception as e:
            st.error(f"Could not solve system: {e}")

else:
    st.subheader(f"{method} Interpolation")
    st.write("Enter x and y values separated by commas. Newton Forward/Backward and Stirling require equally spaced x values.")
    c1, c2, c3 = st.columns(3)
    x_text = c1.text_input("x values", value="0, 1, 2, 3")
    y_text = c2.text_input("y values", value="1, 2, 5, 10")
    x_target = c3.number_input("Target x", value=1.5, format="%.8f")
    if st.button("Interpolate", type="primary"):
        try:
            xs = _parse_float_list(x_text)
            ys = _parse_float_list(y_text)
            if len(xs) != len(ys):
                raise ValueError("x and y must contain the same number of values.")
            if len(xs) < 2:
                raise ValueError("Enter at least two data points.")
            if method == "Newton Forward":
                result, diff, s, terms = newton_forward_interp(xs, ys, float(x_target))
                term_rows = [{"Order": k, "Difference": d, "Coefficient": coeff, "Contribution": contribution} for k, d, coeff, contribution in terms]
                interp_callable = lambda z: newton_forward_interp(xs, ys, float(z))[0]
            elif method == "Newton Backward":
                result, diff, s, terms = newton_backward_interp(xs, ys, float(x_target))
                term_rows = [{"Order": k, "Difference": d, "Coefficient": coeff, "Contribution": contribution} for k, d, coeff, contribution in terms]
                interp_callable = lambda z: newton_backward_interp(xs, ys, float(z))[0]
            elif method == "Stirling":
                result, diff, mid, s, terms = stirling_interp(xs, ys, float(x_target))
                term_rows = [{"Order": t[0], "Difference": t[1], "Term": t[2], "Coefficient": t[3], "Contribution": t[4]} for t in terms]
                interp_callable = lambda z: stirling_interp(xs, ys, float(z))[0]
            else:
                result, terms = lagrange_interp(xs, ys, float(x_target))
                term_rows = [{"Point": k, "xᵢ": xi, "yᵢ": yi, "Lᵢ(x)": li, "Contribution": contribution} for k, xi, yi, li, contribution in terms]
                interp_callable = lambda z: lagrange_interp(xs, ys, float(z))[0]
            st.metric(f"Interpolated value at x = {x_target:g}", f"{result:.10g}")
            show_table(term_rows, list(term_rows[0].keys()) if term_rows else [])
            if method != "Lagrange":
                st.subheader("Finite difference table")
                st.dataframe(pd.DataFrame(diff), use_container_width=True, hide_index=True)
            fig = Figure(figsize=(8, 4.2), facecolor="#0d1117")
            ax = fig.subplots()
            plot_interpolation(ax, xs, ys, float(x_target), float(result), method, interp_callable)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not interpolate: {e}")

st.divider()
st.caption("Educational demo. Numerical results depend on the selected method, input values, and convergence behavior.")
