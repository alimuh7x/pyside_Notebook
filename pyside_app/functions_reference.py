from __future__ import annotations

from utils.notebook_eval import ALLOWED_NOTEBOOK_FUNCTIONS, PHYSICAL_CONSTANTS, UNIT_CONSTANTS


def build_functions_reference_html() -> str:
    names = sorted(set(ALLOWED_NOTEBOOK_FUNCTIONS.keys()) | set(PHYSICAL_CONSTANTS.keys()) | set(UNIT_CONSTANTS.keys()))
    print(f"[debug][functions-reference] build_html count={len(names)}", flush=True)
    documented_sections = [
        (
            "How To Use",
            [
                (
                    "Notebook syntax",
                    "Call functions directly in a code cell. You do not need any prefix for the built-in helpers.",
                    "Example: <code>linspace(0, 1, 5)</code> or <code>plane_stress(210*GPa, 0.3, 0.001, 0.0)</code>",
                ),
                (
                    "Units and constants",
                    "Units are plain numeric multipliers. Physical constants are predefined symbols.",
                    "Example: <code>length_mm = 25 * mm</code>, <code>pressure = 2.5 * MPa</code>, <code>weight = 10 * g_n</code>",
                ),
                (
                    "Arrays",
                    "Most numeric helpers accept Python lists, tuples, or arrays and return NumPy-style results.",
                    "Example: <code>mean([1, 2, 3, 4])</code>, <code>fft(signal)</code>, <code>shape(u)</code>",
                ),
            ],
        ),
        (
            "Math And Arrays",
            [
                ("linspace(start, stop, num=50)", "Create evenly spaced points.", "Example: <code>x = linspace(0, 10, 101)</code>"),
                ("arange(start, stop=None, step=1.0)", "Create regularly spaced values.", "Example: <code>arange(0, 5, 0.5)</code>"),
                ("reshape(v, *shape)", "Reshape a vector or array.", "Example: <code>reshape(arange(0, 9), 3, 3)</code>"),
                ("meshgrid(x, y)", "Build 2D coordinate grids.", "Example: <code>X, Y = meshgrid(linspace(-1,1,50), linspace(-1,1,50))</code>"),
                ("clip(v, lo, hi)", "Limit values to a range.", "Example: <code>clip(signal, 0, 1)</code>"),
                ("normalize(v)", "Return a unit-length vector.", "Example: <code>normalize([3, 4])</code>"),
            ],
        ),
        (
            "Statistics And Fitting",
            [
                ("mean(v), std(v), median(v)", "Basic statistics on vectors or arrays.", "Example: <code>mean(temperature_c)</code>"),
                ("percentile(v, p), quantile(v, q)", "Extract quantiles from data.", "Example: <code>percentile(stress_mpa, 95)</code>"),
                ("polyfit(x, y, deg)", "Fit a polynomial.", "Example: <code>slope, intercept = polyfit(strain, stress_mpa, 1)</code>"),
                ("polyval(p, x)", "Evaluate polynomial coefficients.", "Example: <code>polyval([2, 1], 3)</code>"),
                ("corrcoef(x, y), cov(x, y)", "Correlation and covariance.", "Example: <code>corrcoef(time_s, fraction)</code>"),
            ],
        ),
        (
            "Linear Algebra",
            [
                ("dot(a, b), cross(a, b), norm(v)", "Vector algebra helpers.", "Example: <code>dot([1,2,3], [4,5,6])</code>"),
                ("det(M), inv(M), pinv(M)", "Determinant and matrix inverse helpers.", "Example: <code>det([[1, 2], [3, 4]])</code>"),
                ("solve(A, b), lstsq(A, b)", "Solve linear systems or least-squares systems.", "Example: <code>solve([[2,1],[1,3]], [1,2])</code>"),
                ("eig(M), eig_full(M)", "Eigenvalue helpers.", "Example: <code>eig([[2,0],[0,3]])</code>"),
                ("qr(M), cholesky(M)", "Matrix factorization helpers.", "Example: <code>qr([[1,2],[3,4]])</code>"),
            ],
        ),
        (
            "Calculus And FFT",
            [
                ("gradient(f, dx=1.0)", "Numerical gradient of a 1D signal.", "Example: <code>gradient(u, dx)</code>"),
                ("trapz(y, x=None, dx=1.0)", "Numerical integral using the trapezoid rule.", "Example: <code>trapz(stress_mpa, strain)</code>"),
                ("interp(x, xp, fp)", "1D interpolation.", "Example: <code>interp(0.35, strain, stress_mpa)</code>"),
                ("fft(v), ifft(v), fftfreq(n, dt=1.0)", "Fourier analysis helpers.", "Example: <code>freq = fftfreq(len(signal), dt)</code>"),
            ],
        ),
        (
            "Engineering",
            [
                ("cfl_dt(dx, u, cfl=1.0)", "Stable time-step estimate for advection.", "Example: <code>dt = cfl_dt(dx, velocity, 0.8)</code>"),
                ("diffusion_dt(dx, D, f=0.5)", "Stable explicit time step for diffusion.", "Example: <code>dt = diffusion_dt(dx, diffusivity)</code>"),
                ("fourier_number(D, dt, dx)", "Diffusion Fourier number.", "Example: <code>Fo = fourier_number(D, dt, dx)</code>"),
                ("peclet(u, L, D)", "Péclet number for advection-diffusion.", "Example: <code>Pe = peclet(velocity, length_mm, diffusivity)</code>"),
                ("domain_points(L, dx), cell_size(L, N)", "Grid sizing helpers.", "Example: <code>Nx = domain_points(1.0, 0.02)</code>"),
            ],
        ),
        (
            "Mechanics",
            [
                ("hooke_1d(E, eps)", "1D Hooke's law: stress = E * strain.", "Example: <code>sigma = hooke_1d(210*GPa, 0.001)</code>"),
                ("hooke_3d(E, nu, exx, eyy, ezz, exy=0.0, eyz=0.0, exz=0.0)", "3D isotropic elastic stress vector.", "Example: <code>hooke_3d(210*GPa, 0.3, 0.001, 0, 0)</code>"),
                ("plane_stress(E, nu, exx, eyy, exy=0.0)", "Plane-stress constitutive response.", "Example: <code>plane_stress(210*GPa, 0.3, 0.001, 0.0)</code>"),
                ("plane_strain(E, nu, exx, eyy, exy=0.0)", "Plane-strain constitutive response.", "Example: <code>plane_strain(210*GPa, 0.3, 0.001, 0.0)</code>"),
                ("von_mises(s11, s22, s33, s12=0.0, s23=0.0, s13=0.0)", "Von Mises equivalent stress.", "Example: <code>von_mises(300*MPa, 50*MPa, 0, 10*MPa)</code>"),
            ],
        ),
        (
            "Dimensionless Numbers",
            [
                ("reynolds(rho, u, L, mu)", "Reynolds number.", "Example: <code>Re = reynolds(1000, 2, 0.01, 1e-3)</code>"),
                ("prandtl(mu, cp, k)", "Prandtl number.", "Example: <code>Pr = prandtl(1.8e-5, 1005, 0.026)</code>"),
                ("nusselt_dittus(Re, Pr, heating=True)", "Dittus-Boelter Nusselt correlation.", "Example: <code>Nu = nusselt_dittus(Re, Pr)</code>"),
                ("grashof(g_acc, beta, dT, L, nu)", "Grashof number for buoyancy-driven flow.", "Example: <code>Gr = grashof(g_n, 1/300, 50, 0.1, 1.5e-5)</code>"),
                ("biot(h, L, k)", "Biot number.", "Example: <code>Bi = biot(25, 0.01, 15)</code>"),
            ],
        ),
        (
            "Composition And Units",
            [
                ("wt_to_mol(wt, M), mol_to_wt(x, M)", "Convert between weight and mole composition.", "Example: <code>mol_to_wt([0.7, 0.3], [55.845, 12.011])</code>"),
                ("wt_to_mol2(wt_B, M_A, M_B)", "Binary alloy scan helper.", "Example: <code>wt_to_mol2(linspace(0, 100, 11), 55.845, 12.011)</code>"),
                ("Useful units", "Predefined units are numeric scalers.", "Example: <code>5 * mm</code>, <code>2.5 * MPa</code>, <code>10 * kW</code>, <code>1.5 * hour</code>"),
                ("Useful constants", "Predefined physical constants are also available.", "Example: <code>g_n</code>, <code>R_gas</code>, <code>k_B</code>, <code>N_A</code>, <code>atm</code>"),
            ],
        ),
    ]

    section_html = []
    for title, entries in documented_sections:
        print(f"[debug][functions-reference] section={title!r} entries={len(entries)}", flush=True)
        rows = "".join(
            "<tr>"
            f"<td style='padding:8px 10px; border-bottom:1px solid #e2e8f0; vertical-align:top; width:32%;'><code>{name}</code></td>"
            f"<td style='padding:8px 10px; border-bottom:1px solid #e2e8f0; vertical-align:top; color:#334155; width:34%;'>{description}</td>"
            f"<td style='padding:8px 10px; border-bottom:1px solid #e2e8f0; vertical-align:top; color:#64748b; width:34%;'>{example}</td>"
            "</tr>"
            for name, description, example in entries
        )
        section_html.append(
            "<h4 style='margin-bottom:8px;'>"
            f"{title}"
            "</h4>"
            "<table style='width:100%; border-collapse:collapse; margin-bottom:18px;'>"
            "<thead>"
            "<tr>"
            "<th style='text-align:left; padding:8px 10px; border-bottom:2px solid #cbd5e1; color:#0f172a;'>Function</th>"
            "<th style='text-align:left; padding:8px 10px; border-bottom:2px solid #cbd5e1; color:#0f172a;'>What It Does</th>"
            "<th style='text-align:left; padding:8px 10px; border-bottom:2px solid #cbd5e1; color:#0f172a;'>Example</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    return (
        "<h3>Available Functions</h3>"
        "<p>This panel shows the most important notebook helpers, what they do, and example calls you can paste into a cell.</p>"
        + "".join(section_html)
    )
