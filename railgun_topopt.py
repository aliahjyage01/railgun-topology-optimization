"""
Optimisation topologique d'un Rail Gun - V4 (FINALE)
=====================================================
Version pragmatique et propre :
- Modélisation : 1 rail, l'autre par symétrie
- Objectif principal : ∫(J·n)² sur la sortie (V2 simple)
- Objectif pondéré : ∫ w(y) σ |∇V|² avec w(y) ∝ 1/(distance au rail opposé)²
  → encode la physique de Biot-Savart de manière LOCALE
- Étude de sensibilité au volume (pour justifier le choix de V_FRAC)
- Continuation SIMP qui marche
- Learning rate normalisé
"""

from ngsolve import *
from netgen.occ import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# ============================================================
# 1. PARAMÈTRES
# ============================================================

L_RAIL = 100.0
W_RAIL = 10.0
GAP = 20.0  # distance entre faces internes des deux rails

SIGMA_CU = 1.0
SIGMA_MIN = 1e-4

V_IN = 1.0

VOL_FRAC = 0.5  # valeur par défaut, l'étude de sensibilité va explorer
N_ITER = 80
LR_INIT = 0.5
FILTER_RADIUS = 2.5
MESH_SIZE = 1.5

P_INIT = 1.0
P_FINAL = 4.0
P_INCREASE_EVERY = 12

# Choix de l'objectif
USE_WEIGHTED_OBJECTIVE = True   # True = pondéré (proxy F_x), False = simple (V2)


# ============================================================
# 2. GÉOMÉTRIE
# ============================================================

def make_mesh():
    """Rail haut : y ∈ [0, W_RAIL], face interne (vers le rail bas) à y=0."""
    rail = MoveTo(0, 0).Rectangle(L_RAIL, W_RAIL).Face()
    rail.faces.name = "domain"
    for edge in rail.edges:
        cx = edge.center.x
        cy = edge.center.y
        if abs(cx) < 0.1:
            edge.name = "gamma_in"
        elif abs(cx - L_RAIL) < 0.1:
            edge.name = "gamma_out"
        else:
            edge.name = "wall"
    geo = OCCGeometry(rail, dim=2)
    return Mesh(geo.GenerateMesh(maxh=MESH_SIZE))


# ============================================================
# 3. SOLVEURS
# ============================================================

def sigma_of_rho(rho_gf, p):
    return SIGMA_MIN + (SIGMA_CU - SIGMA_MIN) * rho_gf**p


def solve_state(rho_gf, mesh, fes, p):
    V = GridFunction(fes)
    V.Set(V_IN, definedon=mesh.Boundaries("gamma_in"))
    sigma = sigma_of_rho(rho_gf, p)
    u, v = fes.TnT()
    a = BilinearForm(fes, symmetric=True)
    a += sigma * grad(u) * grad(v) * dx
    a.Assemble()
    f = LinearForm(fes)
    f.Assemble()
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * V.vec
    V.vec.data += a.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky") * r
    return V, sigma


def helmholtz_filter(rho_raw, fes_rho, r_min):
    rho_filt = GridFunction(fes_rho)
    u, v = fes_rho.TnT()
    a = BilinearForm(fes_rho, symmetric=True)
    a += (r_min**2 * grad(u) * grad(v) + u * v) * dx
    a.Assemble()
    f = LinearForm(fes_rho)
    f += rho_raw * v * dx
    f.Assemble()
    rho_filt.vec.data = a.mat.Inverse(inverse="sparsecholesky") * f.vec
    return rho_filt


# ============================================================
# 4. POIDS w(y) — APPROXIMATION DE BIOT-SAVART
# ============================================================
# Le rail bas est centré en y_bot = -GAP/2 - W_RAIL/2 (relatif au rail haut placé à y=0)
# Distance d'un point (x, y) du rail haut au rail bas (centre) :
#   d(y) = y - y_bot = y + GAP/2 + W_RAIL/2
# Plus on est proche du bord intérieur (y=0, proche du rail bas), plus on contribue à B.
# Poids : w(y) = 1 / (d(y))²  (loi de Biot-Savart locale)

def get_weight():
    """Coefficient function pour le poids w(y)."""
    y_bot_center = -GAP/2 - W_RAIL/2
    d = y - y_bot_center  # distance au centre du rail opposé
    return 1.0 / (d**2 + 1e-6)  # régularisation pour éviter division par zéro


# ============================================================
# 5. OBJECTIF + GRADIENT
# ============================================================

def objective_simple(V, sigma, mesh):
    """Objectif simple V2 : ∫(σ ∂V/∂x)² sur la sortie."""
    n = specialcf.normal(2)
    flux = sigma * grad(V) * n
    return -Integrate(flux**2 * ds(definedon=mesh.Boundaries("gamma_out")), mesh)


def objective_weighted(V, sigma, mesh):
    """Objectif pondéré : ∫ w(y) σ |∇V|² dΩ (proxy de F_x)."""
    w = get_weight()
    return -Integrate(w * sigma * (grad(V) * grad(V)), mesh)


def compute_gradient_simple(V, rho_gf, sigma, mesh, fes_V, fes_rho, p):
    """Gradient pour objectif simple : adjoint complet."""
    n = specialcf.normal(2)
    flux = sigma * grad(V) * n
    
    lam = GridFunction(fes_V)
    lam.Set(0.0, definedon=mesh.Boundaries("gamma_in|gamma_out"))
    
    u, w = fes_V.TnT()
    a_adj = BilinearForm(fes_V, symmetric=True)
    a_adj += sigma * grad(u) * grad(w) * dx
    a_adj.Assemble()
    
    f_adj = LinearForm(fes_V)
    f_adj += 2.0 * flux * (sigma * grad(w) * n) * ds(definedon=mesh.Boundaries("gamma_out"))
    f_adj.Assemble()
    
    r = f_adj.vec.CreateVector()
    r.data = f_adj.vec - a_adj.mat * lam.vec
    lam.vec.data += a_adj.mat.Inverse(freedofs=fes_V.FreeDofs(), inverse="sparsecholesky") * r
    
    dsigma_drho = p * (SIGMA_CU - SIGMA_MIN) * rho_gf**(p - 1 + 1e-9)
    grad_expr = dsigma_drho * (grad(V) * grad(lam))
    
    return _project_to_rho_space(grad_expr, fes_rho)


def compute_gradient_weighted(V, rho_gf, sigma, mesh, fes_V, fes_rho, p):
    """Gradient pour objectif pondéré.
    J = -∫ w σ |∇V|² . dJ/dV(δV) = -2 ∫ w σ ∇V·∇δV.
    Adjoint : ∫ σ ∇λ·∇δV = -dJ/dV(δV) = +2 ∫ w σ ∇V·∇δV  pour tout δV.
    Et : dJ/dρ = -∂σ/∂ρ · w · |∇V|² + ∂σ/∂ρ · ∇V·∇λ
    """
    w_weight = get_weight()
    
    # Adjoint
    lam = GridFunction(fes_V)
    lam.Set(0.0, definedon=mesh.Boundaries("gamma_in|gamma_out"))
    
    u, wt = fes_V.TnT()
    a_adj = BilinearForm(fes_V, symmetric=True)
    a_adj += sigma * grad(u) * grad(wt) * dx
    a_adj.Assemble()
    
    f_adj = LinearForm(fes_V)
    f_adj += 2.0 * w_weight * sigma * grad(V) * grad(wt) * dx
    f_adj.Assemble()
    
    r = f_adj.vec.CreateVector()
    r.data = f_adj.vec - a_adj.mat * lam.vec
    lam.vec.data += a_adj.mat.Inverse(freedofs=fes_V.FreeDofs(), inverse="sparsecholesky") * r
    
    dsigma_drho = p * (SIGMA_CU - SIGMA_MIN) * rho_gf**(p - 1 + 1e-9)
    # dJ/dρ : direct + adjoint
    grad_expr = dsigma_drho * (-w_weight * (grad(V) * grad(V)) + (grad(V) * grad(lam)))
    
    return _project_to_rho_space(grad_expr, fes_rho)


def _project_to_rho_space(grad_expr, fes_rho):
    """Projette une expression sur l'espace de ρ via mass matrix."""
    grad_gf = GridFunction(fes_rho)
    trial, test = fes_rho.TnT()
    M = BilinearForm(fes_rho, symmetric=True)
    M += trial * test * dx
    M.Assemble()
    rhs = LinearForm(fes_rho)
    rhs += grad_expr * test * dx
    rhs.Assemble()
    grad_gf.vec.data = M.mat.Inverse(inverse="sparsecholesky") * rhs.vec
    return grad_gf


def project_volume(rho_gf, mesh, vol_target):
    rho_arr = np.array(rho_gf.vec.FV().NumPy())
    def vol_for(lam):
        rho_gf.vec.FV().NumPy()[:] = np.clip(rho_arr - lam, 0.0, 1.0)
        return Integrate(rho_gf, mesh)
    lo, hi = -1.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if vol_for(mid) > vol_target:
            lo = mid
        else:
            hi = mid
    rho_gf.vec.FV().NumPy()[:] = np.clip(rho_arr - mid, 0.0, 1.0)


# ============================================================
# 6. BOUCLE D'OPTIM (réutilisable pour étude de sensibilité)
# ============================================================

def optimize(mesh, vol_frac, n_iter=N_ITER, verbose=True):
    fes_V = H1(mesh, order=1, dirichlet="gamma_in|gamma_out")
    fes_rho = H1(mesh, order=1)
    
    rho = GridFunction(fes_rho)
    rho.Set(vol_frac)
    
    full_volume = Integrate(CoefficientFunction(1.0), mesh)
    vol_target = vol_frac * full_volume
    
    if USE_WEIGHTED_OBJECTIVE:
        obj_fn = objective_weighted
        grad_fn = compute_gradient_weighted
    else:
        obj_fn = objective_simple
        grad_fn = compute_gradient_simple
    
    # État initial
    V_init, sigma_init = solve_state(rho, mesh, fes_V, P_FINAL)
    J_init = obj_fn(V_init, sigma_init, mesh)
    
    history = [J_init]
    p_simp = P_INIT
    lr = LR_INIT
    
    for it in range(n_iter):
        if it > 0 and it % P_INCREASE_EVERY == 0:
            p_simp = min(p_simp + 0.5, P_FINAL)
        
        rho_filt = helmholtz_filter(rho, fes_rho, FILTER_RADIUS)
        V, sigma = solve_state(rho_filt, mesh, fes_V, p_simp)
        J_val = obj_fn(V, sigma, mesh)
        grad_J = grad_fn(V, rho_filt, sigma, mesh, fes_V, fes_rho, p_simp)
        
        grad_norm = np.linalg.norm(np.array(grad_J.vec.FV().NumPy()))
        if grad_norm > 1e-12:
            step = lr / grad_norm
        else:
            step = lr
        rho.vec.data -= step * grad_J.vec
        project_volume(rho, mesh, vol_target)
        
        history.append(J_val)
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            rho_arr = np.array(rho.vec.FV().NumPy())
            n_grey = np.sum((rho_arr > 0.05) & (rho_arr < 0.95))
            print(f"  Iter {it:3d} | p={p_simp:.1f} | J={abs(J_val):.4e} | "
                  f"gris={100*n_grey/len(rho_arr):.1f}%", flush=True)
    
    # État final
    rho_filt = helmholtz_filter(rho, fes_rho, FILTER_RADIUS)
    V_final, sigma_final = solve_state(rho_filt, mesh, fes_V, P_FINAL)
    J_final = obj_fn(V_final, sigma_final, mesh)
    
    return rho_filt, V_final, sigma_final, J_init, J_final, history


# ============================================================
# 7. ÉTUDE DE SENSIBILITÉ AU VOLUME
# ============================================================

def sensitivity_study(mesh):
    """Balaie V_FRAC = 0.3, 0.4, ..., 0.8 et trace J_final / J_uniform."""
    print("\n" + "=" * 60, flush=True)
    print("ÉTUDE DE SENSIBILITÉ AU VOLUME", flush=True)
    print("=" * 60, flush=True)
    
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    
    for vf in fractions:
        print(f"\nOptim avec V_FRAC = {vf}...", flush=True)
        _, _, _, J_init, J_final, _ = optimize(mesh, vf, n_iter=40, verbose=False)
        gain = abs(J_final) / abs(J_init)
        results.append((vf, abs(J_init), abs(J_final), gain))
        print(f"  J_init={abs(J_init):.3e} | J_final={abs(J_final):.3e} | gain={gain:.2f}x",
              flush=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fracs = [r[0] for r in results]
    finals = [r[2] for r in results]
    gains = [r[3] for r in results]
    
    ax1.plot(fracs, finals, 'b-o', markersize=8)
    ax1.set_xlabel("Fraction volumique")
    ax1.set_ylabel("Objectif final |J|")
    ax1.set_title("Objectif optimisé en fonction du volume disponible")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(fracs, gains, 'r-s', markersize=8)
    ax2.set_xlabel("Fraction volumique")
    ax2.set_ylabel("Gain (J_final / J_initial)")
    ax2.set_title("Gain relatif de l'optimisation")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1.0, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("etude_volume.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  → etude_volume.png", flush=True)
    
    # Trouver le meilleur compromis
    best_idx = np.argmax(gains)
    print(f"\n>>> Meilleur gain à V_FRAC = {fractions[best_idx]:.2f} : {gains[best_idx]:.2f}x",
          flush=True)
    return fractions[best_idx]


# ============================================================
# 8. VISU
# ============================================================

def extract_mesh_arrays(mesh):
    pts = np.array([[v.point[0], v.point[1]] for v in mesh.vertices])
    tris = []
    for el in mesh.Elements(VOL):
        verts = [v.nr for v in el.vertices]
        if len(verts) == 3:
            tris.append(verts)
    return pts, np.array(tris)


def evaluate_cf_at_vertices(cf, mesh):
    vals = np.zeros(mesh.nv)
    for i, v in enumerate(mesh.vertices):
        p = v.point
        mp = mesh(p[0], p[1])
        vals[i] = cf(mp)
    return vals


def plot_two_rails(pts, tris, vals, title, filename, cmap="viridis",
                   vmin=None, vmax=None, label=""):
    pts_top = pts.copy()
    pts_top[:, 1] += GAP/2
    pts_bot = pts.copy()
    pts_bot[:, 1] = -pts_bot[:, 1] - GAP/2

    triang_top = Triangulation(pts_top[:, 0], pts_top[:, 1], tris)
    triang_bot = Triangulation(pts_bot[:, 0], pts_bot[:, 1], tris)

    fig, ax = plt.subplots(figsize=(11, 5))
    if vmin is None: vmin = vals.min()
    if vmax is None: vmax = vals.max()
    tcf = ax.tricontourf(triang_top, vals, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.tricontourf(triang_bot, vals, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axvspan(L_RAIL, L_RAIL + 8, ymin=0.3, ymax=0.7,
               alpha=0.3, color='gray', label='projectile')
    plt.colorbar(tcf, ax=ax, label=label or title)
    ax.set_aspect('equal')
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {filename}", flush=True)


def plot_history(history, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = np.arange(len(history))
    ax.plot(iters, np.abs(history), 'b-', linewidth=1.5)
    ax.set_xlabel("Itération")
    ax.set_ylabel("|J|")
    ax.set_title("Convergence de l'optimisation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {filename}", flush=True)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("=" * 60, flush=True)
    print("Rail Gun V4 - Optimisation topologique", flush=True)
    print(f"Objectif : {'PONDÉRÉ (proxy F_x)' if USE_WEIGHTED_OBJECTIVE else 'simple flux²'}",
          flush=True)
    print("=" * 60, flush=True)
    
    mesh = make_mesh()
    print(f"Maillage : {mesh.ne} éléments, {mesh.nv} sommets", flush=True)
    
    # Étude de sensibilité
    best_vf = sensitivity_study(mesh)
    
    # Optim finale avec le meilleur V_FRAC
    print("\n" + "=" * 60, flush=True)
    print(f"OPTIMISATION FINALE — V_FRAC = {best_vf}", flush=True)
    print("=" * 60, flush=True)
    rho_final, V_final, sigma_final, J_init, J_final, history = optimize(
        mesh, best_vf, n_iter=N_ITER, verbose=True
    )
    
    print(f"\n{'='*60}", flush=True)
    print(f"RÉSULTATS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"V_FRAC adopté : {best_vf}", flush=True)
    print(f"J initial     : {abs(J_init):.6e}", flush=True)
    print(f"J final       : {abs(J_final):.6e}", flush=True)
    print(f"GAIN          : {abs(J_final/J_init):.2f}x", flush=True)
    
    # Pour comparaison : état uniforme à V_FRAC
    fes_V = H1(mesh, order=1, dirichlet="gamma_in|gamma_out")
    fes_rho = H1(mesh, order=1)
    rho_uni = GridFunction(fes_rho)
    rho_uni.Set(best_vf)
    V_uni, sigma_uni = solve_state(rho_uni, mesh, fes_V, P_FINAL)
    
    # Figures
    print("\nFigures...", flush=True)
    pts, tris = extract_mesh_arrays(mesh)
    
    rho_vals = evaluate_cf_at_vertices(rho_final, mesh)
    V_vals = evaluate_cf_at_vertices(V_final, mesh)
    j_norm_cf = sigma_final * sqrt(grad(V_final) * grad(V_final) + 1e-12)
    j_vals = evaluate_cf_at_vertices(j_norm_cf, mesh)
    
    j_uni_cf = sigma_uni * sqrt(grad(V_uni) * grad(V_uni) + 1e-12)
    j_uni_vals = evaluate_cf_at_vertices(j_uni_cf, mesh)
    
    j_max = max(j_vals.max(), j_uni_vals.max())
    
    plot_two_rails(pts, tris, rho_vals, f"ρ - topologie optimisée (V_FRAC={best_vf})",
                   "rho_optimal.png", cmap="binary_r", vmin=0, vmax=1,
                   label="ρ (1=cuivre, 0=trou)")
    plot_two_rails(pts, tris, V_vals, "Potentiel V (rail optimisé)",
                   "potentiel.png", cmap="RdBu_r", label="V")
    plot_two_rails(pts, tris, j_uni_vals, f"‖J‖ avant optim (ρ={best_vf} uniforme)",
                   "courant_initial.png", cmap="hot", vmin=0, vmax=j_max, label="‖J‖")
    plot_two_rails(pts, tris, j_vals, "‖J‖ après optimisation topologique",
                   "courant_final.png", cmap="hot", vmin=0, vmax=j_max, label="‖J‖")
    plot_history(history, "convergence.png")
    
    # Plot du poids w(y) pour la présentation
    if USE_WEIGHTED_OBJECTIVE:
        fig, ax = plt.subplots(figsize=(7, 5))
        y_arr = np.linspace(0, W_RAIL, 100)
        y_bot = -GAP/2 - W_RAIL/2
        d = y_arr - y_bot
        w_vals = 1.0 / d**2
        ax.plot(y_arr, w_vals / w_vals.max(), 'b-', linewidth=2)
        ax.set_xlabel("y dans le rail (mm)")
        ax.set_ylabel("w(y) normalisé")
        ax.set_title("Poids w(y) ∝ 1/d² : approximation locale de Biot-Savart\n"
                     "Le courant proche du rail opposé contribue plus à F_Laplace")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color='r', linestyle='--', alpha=0.5,
                   label='Bord intérieur (proche du rail opposé)')
        ax.axvline(W_RAIL, color='g', linestyle='--', alpha=0.5,
                   label='Bord extérieur')
        ax.legend()
        plt.tight_layout()
        plt.savefig("ponderation.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  → ponderation.png", flush=True)
    
    print("\nTerminé !", flush=True)


if __name__ == "__main__":
    main()
