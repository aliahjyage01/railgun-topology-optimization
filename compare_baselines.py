"""
Le gain annonce par railgun_topopt.py est-il un vrai gain ?
============================================================
railgun_topopt.py compare le design optimise a un rail de densite uniforme
rho = VOL_FRAC. Avec la penalisation SIMP (p = 4), ce rail "gris" a une
conductivite de VOL_FRAC^4 : c'est une reference non physique, et le gain
mesure contre elle est surtout un effet de la penalisation.

Ce script compare, a VOLUME DE CUIVRE EGAL et avec le meme objectif J :
  - uniforme    : rho = VOL_FRAC partout (la reference du script principal)
  - optimise    : sortie brute de l'optimiseur (80 iterations), grise
  - seuille     : sortie de l'optimiseur seuillee en 0/1 au meme volume
  - bande int.  : cuivre plein dans une bande le long du bord interieur
                  (cote rail oppose), dessinee a la main
  - bande ext.  : la meme bande le long du bord exterieur

Toutes les densites sont evaluees avec p = P_FINAL, comme dans le script
principal. Le volume des designs 0/1 est ajuste par bissection.

Lancer :  python compare_baselines.py      (~10 min)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from ngsolve import H1, GridFunction, Integrate, CoefficientFunction

import railgun_topopt as R

FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_ITER = 80


def main():
    mesh = R.make_mesh()
    fes_V = H1(mesh, order=1, dirichlet="gamma_in|gamma_out")
    fes_rho = H1(mesh, order=1)
    volume_total = Integrate(CoefficientFunction(1.0), mesh)
    y_sommets = np.array([v.point[1] for v in mesh.vertices])

    def evalue(rho_sommets):
        """Objectif J (valeur absolue) et fraction volumique d'une densite."""
        rho = GridFunction(fes_rho)
        rho.vec.FV().NumPy()[:] = np.clip(rho_sommets, 0.0, 1.0)
        V, sigma = R.solve_state(rho, mesh, fes_V, R.P_FINAL)
        return abs(R.objective_weighted(V, sigma, mesh)), Integrate(rho, mesh) / volume_total

    def bande(largeur, interieur):
        """Bande pleine le long d'un bord, transition lineaire de 1 mm (P1)."""
        d = y_sommets if interieur else R.W_RAIL - y_sommets
        return np.clip(largeur - d + 0.5, 0.0, 1.0)

    def ajuste_volume(fabrique, cible, lo, hi, croissant=True):
        """Bissection sur le parametre de fabrique pour atteindre le volume cible."""
        for _ in range(40):
            m = 0.5 * (lo + hi)
            _, v = evalue(fabrique(m))
            if (v < cible) == croissant:
                lo = m
            else:
                hi = m
        return evalue(fabrique(0.5 * (lo + hi)))

    J_plein, _ = evalue(np.ones(len(y_sommets)))
    print(f"Rail plein (rho = 1, 100 % de cuivre) : J = {J_plein:.3e}\n")
    print("Fraction de J_plein conservee : (seuille) et (bande ext) en fin de ligne\n")

    print(f"{'V_FRAC':>6s} {'uniforme':>10s} {'optimise':>10s} {'seuille':>10s} "
          f"{'bande int':>10s} {'bande ext':>10s} {'seuil/int':>9s} {'int/ext':>8s} "
          f"{'seuil/plein':>11s} {'ext/plein':>9s}")
    for vf in FRACTIONS:
        J_uni, _ = evalue(np.full(len(y_sommets), vf))

        rho_opt, _, _, _, J_opt, _ = R.optimize(mesh, vf, n_iter=N_ITER, verbose=False)
        a = np.array(rho_opt.vec.FV().NumPy())
        J_opt = abs(J_opt)

        # seuil t : volume decroissant quand t augmente
        J_seuil, _ = ajuste_volume(lambda t: (a >= t).astype(float), vf, 0.0, 1.0,
                                   croissant=False)
        J_int, _ = ajuste_volume(lambda w: bande(w, True), vf, 0.0, R.W_RAIL)
        J_ext, _ = ajuste_volume(lambda w: bande(w, False), vf, 0.0, R.W_RAIL)

        print(f"{vf:6.1f} {J_uni:10.3e} {J_opt:10.3e} {J_seuil:10.3e} "
              f"{J_int:10.3e} {J_ext:10.3e} {J_seuil / J_int:9.2f} {J_int / J_ext:8.2f} "
              f"{J_seuil / J_plein:11.2f} {J_ext / J_plein:9.2f}",
              flush=True)


if __name__ == "__main__":
    main()
