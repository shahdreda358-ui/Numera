# Numerical Methods Lab — Website

A responsive, browser-based numerical methods calculator designed for GitHub Pages.

## Features
- Root finding: Bisection, False Position, Newton–Raphson, Secant
- Linear systems: Gauss–Seidel, Jacobi
- Interpolation: Newton Forward, Newton Backward, Stirling (central-difference table), Lagrange
- Step-by-step result tables
- Function/interpolation charts
- Export iteration tables as CSV
- Responsive dark-blue UI

## Run locally
Open `index.html` in a browser while connected to the internet (Math.js and Chart.js are loaded from CDN).

## Publish with GitHub Pages
1. Create a GitHub repository, e.g. `Numerical-Methods`.
2. Upload `index.html`, `style.css`, `script.js`, and optionally this README to the repository root.
3. Open **Settings → Pages**.
4. Under **Build and deployment**, choose **Deploy from a branch**.
5. Choose branch **main** and folder **/(root)**, then Save.
6. Wait for the Pages URL to appear. The root must contain the file named exactly `index.html`.

## Input notes
- Function syntax uses `x`, for example `x^3 - 4*x - 9`; multiplication must be explicit (`4*x`).
- Matrix A: one row per line; numbers separated by commas or spaces.
- Interpolation x and y values must be entered in matching order and be the same length.
- Newton Forward, Newton Backward, and Stirling require equally spaced x-values.
- Numerical methods can fail to converge for some starting values or functions. Check the function, interval, initial guesses, and tolerance.
