#!/usr/bin/env python3
"""Builds claims.json (the registry of every number in the paper) and proposes numbers_allowlist.json.

Run once by the builder; afterwards claims.json is the frozen registry and is edited only with a history record.
    python3 tools/build_claims.py --config config_cloud.json
Values are computed from the declared recipe / data / literature source at build time. Claims that depend on the
v7 reruns in progress are status "pending_rerun" (value null) with their producing script and an extract spec.
Claims whose current source is v6 data (pre-fix engine) are "provisional" with the v7 successor recorded.
"""
import os, sys, re, json, math, argparse, datetime, hashlib
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("lib", "tests", "adapters"):
    sys.path.insert(0, os.path.join(HERE, sub))
sys.path.insert(0, HERE)
import vcore, texscan, claims_recipes as CR
from vcore import Ctx, load_config, sha256_file
import prov

M, I, T, R, D, A = ("main.tex", "sections/introduction.tex", "sections/theory.tex", "sections/results.tex",
                    "sections/discussion.tex", "sections/appendix.tex")
MATS = ["MoS2", "MoSe2", "WS2", "WSe2"]
CLAIMS = []
SPECS = []


def cl(id, file, near, printed, kind, nth=0, **kw):
    SPECS.append(dict(id=id, file=file, near=near, printed=printed, kind=kind, nth=nth, **kw))


# ================================================================================================ registry of the FINAL text
# Paper: stage-2 source (live Overleaf + 82 audit edits + 29 v7 edits, 2026-10-14). Values come from the v7 outputs
# (v7_out: prod7/edge7/chiwR9v7/lam7/decomp7; v7_checks: c2/c3), the reference (ref/refimpl.py), the literature tables,
# kp2 recomputes, or the producing scripts' settings (method_param + G6 regex). Definitions (final text):
#   van Hove energy = omega of max |eta(60 T)| (prod7); van Hove chi = eta(1 T)/(1 T); "within 5%" and "below the linear
#   extrapolation" relative to that initial slope; threshold chi = edge7 chi_SI_per_T (fit s B + c B^3).
V7 = "v7jobs"
SI_T = "A m V^-2 s^-1 T^-1"
EDGE_TESTS = ["R4", "R5", "H1", "S1", "C2", "C3", "C6"]
MAC_ONLY = "data-checked on the Mac (G1 in full mode): the cloud copy of edge7_*_TNN_LDA.json is partial"


def prod(script, out=None, **kw):
    d = dict(script=f"{V7}/{script}" if "/" not in script else script)
    if out:
        d["output"] = out
    d.update(kw)
    return d


# ------------------------------------------------------------------------------------------------ ABSTRACT (main.tex)
cl("vH.chi.range.min", M, "field coefficient of $(3$", "3", "range_min", recipe="vh7_slope_range", args=dict(agg="min"),
   display=dict(scale=1e-2, decimals=0), units=SI_T, quantity="smallest van Hove chi = eta(1 T)/1 T (TNN GGA F, prod7) over the four materials",
   producer=prod("maps_v7.py", "prod7_{m}_F.npz"), tests=["S2", "R4"])
cl("vH.chi.range.max", M, "field coefficient of $(3$--$7)", "7", "range_max", recipe="vh7_slope_range", args=dict(agg="max"),
   display=dict(scale=1e-2, decimals=0), units=SI_T, quantity="largest van Hove chi", producer=prod("maps_v7.py", "prod7_{m}_F.npz"))
cl("kp.fp.order.abstract", M, "positive threshold response of about $10^{-3}", "10", "assertion", recipe="kp_fp_order",
   display=dict(mode="order", exponent=-3), quantity="decimal order of the first-principles k.p threshold chi (all four)",
   source=dict(data="kp_out:kp_fp_all_v7.json"), producer=dict(script="kp/kp_fp_all_v7.py"), tests=["S6", "L4", "L5"])

# ------------------------------------------------------------------------------------------------ THEORY
cl("const.gs.theory", T, "and $g_s=2$. The second", "2", "method_param", value=2.0, display=dict(decimals=0),
   quantity="free-electron spin g-factor", check=dict(file="pub_pipeline/tb_core.py", regex=r"G_S\s*=\s*([0-9.]+)"))
cl("moment.K.min", T, "itinerant moment has magnitude $2.6", "2.6", "range_min", recipe="moment_K_range", args=dict(agg="min"),
   display=dict(decimals=1), units="mu_B", quantity="smallest |M_n| at K, K' (v and c, TNN GGA, active spin)", tests=["R2"])
cl("moment.K.max", T, "itinerant moment has magnitude $2.6$--$5.9", "5.9", "range_max", recipe="moment_K_range", args=dict(agg="max"),
   display=dict(decimals=1), units="mu_B", quantity="largest |M_n| at K, K'")
cl("D.max60.theory", T, "factor changes the weight by less than $3", "3", "assertion", recipe="D_max", args=dict(B=60.0),
   display=dict(scale=0.01, mode="less_than"), quantity="max |D_n - 1| at 60 T within 0.3/a of K, K' (v,c, TNN)", tests=["R2"])
cl("D.max60.theory.B", T, "weight by less than $3\\%$ at $\\SI{60}", "60", "method_param", value=60.0, display=dict(decimals=0), units="T",
   quantity="field of the phase-space statement")
cl("D.chi.shift.min", T, "Zeeman-only model it shifts $\\chi$ by $21", "21", "range_min", recipe="decomp7_ratio_range",
   args=dict(where="edge", num="D", den="Z", agg="min", minus_one=True), display=dict(scale=0.01, decimals=0),
   quantity="(chi_D/chi_Z - 1) at E_g+0.1 eV (decomp7 edge, R=0.9/a, h=0.002/a), min over materials", producer=prod("decomp_v7.py", "decomp7_{m}.json"))
cl("D.chi.shift.max", T, "Zeeman-only model it shifts $\\chi$ by $21$--$31", "31", "range_max", recipe="decomp7_ratio_range",
   args=dict(where="edge", num="D", den="Z", agg="max", minus_one=True), display=dict(scale=0.01, decimals=0),
   quantity="(chi_D/chi_Z - 1) at E_g+0.1 eV, max", producer=prod("decomp_v7.py", "decomp7_{m}.json"))
cl("param.edge01.theory", T, "$21$--$31\\%$ at $E_g+0.1", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV",
   check=dict(file="v7jobs/decomp_v7.py", regex=r"edge_eV=Eg\+([0-9.]+)"))
F5 = "v6_code:runs/f5_extended_omega.py"           # pathkey:file (outside code_root)
for n_, near in enumerate(["($\\mu_B B = \\SI{80}{meV}$, or", "($\\mu_B B = \\SI{80}{meV}$, $B \\sim", "($\\mu_B B = \\SI{80}{meV}$, with the energy"]):
    cl(f"fig.zeeman80.{n_}", T, near, "80", "method_param", value=0.08, display=dict(scale=1e-3, decimals=0), units="eV",
       quantity="enlarged Zeeman energy mu_B B of Figs. 1-5", check=dict(file=F5, regex=r"for b in \(0\.0, ([0-9.]+)\)"),
       notes="v6 figure data (f5_extended_omega.py for Fig. 5; fig_theory.py for Figs. 1-4)")
for n_, near in enumerate(["or $B \\sim \\SI{1.4e3}", "$B \\sim \\SI{1.4e3}{\\tesla}$); grey"]):
    cl(f"fig.zeeman80.B.{n_}", T, near, "1.4e3", "derived", recipe="B_for_zeeman", args=dict(muBB_eV=0.08),
       display=dict(mode="sci", decimals=1), units="T", quantity="B = 80 meV / mu_B")
cl("range.Bmax.theory", T, "below use $B \\lesssim \\SI{100}", "100", "method_param", value=100.0, display=dict(decimals=0), units="T",
   quantity="upper end of the field range considered (largest computed field is 60 T)",
   check=dict(file="v7jobs/edge_v7.py", regex=r"np\.arange\(0, ([0-9.]+) \+ 1e-9, DB\)", relation="<="))
cl("lit.lam.MoS2.fig1", T, "coupling $\\lambda = \\SI{0.073}", "0.073", "literature", recipe="lit", args=dict(file="liu_tables.json", path=["lambda_eV", "MoS2"]),
   display=dict(decimals=3), units="eV", lit=dict(ref="Liu PRB 88, 085433", table="IV", column="lambda (GGA)"), tests=["L1", "L3"])
cl("fig5.N", T, "itinerant moment; $N=240", "240", "method_param", value=240.0, display=dict(decimals=0),
   quantity="BZ mesh of the Fig. 5 spectra (v6 data f5_{m}_240.npz, not rerun in v7)", source=dict(data="data_v6:f5_MoS2_240.npz"),
   notes="producer paper/v6_work/code/runs/f5_extended_omega.py (H_Z + moment energy shift + D_v; audit A5); N is argv, fixed by the data file name")
cl("fig5.nu", T, "$N=240$, $\\nu=30", "30", "method_param", value=0.03, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v6_jobs:f5/tb_core.py", regex=r"\nNU\s*=\s*([0-9.]+)"), notes="response.NU = tb.NU of the v6 engine copy (audit A5)")
cl("param.thr.offset.tab2", T, "at $\\hbar\\omega = E_g + 0.01$~eV in the TNN and NN", "0.01", "method_param", value=0.01, display=dict(decimals=2),
   units="eV", check=dict(file="v7jobs/edge_v7.py", regex=r"om = Eg \+ ([0-9.]+) \+"))
for i, m in enumerate(MATS):
    row = {"MoS2": "$\\mathrm{MoS_2}$  & $4.93", "MoSe2": "$\\mathrm{MoSe_2}$ & $4.30", "WS2": "$\\mathrm{WS_2}$   & $5.90",
           "WSe2": "$\\mathrm{WSe_2}$  & $5.21"}[m]
    pr = {"MoS2": ("4.93", "-0.24", "-0.04", "-1.45"), "MoSe2": ("4.30", "-0.02", "+0.03", "-1.23"),
          "WS2": ("5.90", "-0.02", "-0.51", "-1.09"), "WSe2": ("5.21", "+0.08", "-0.26", "-0.93")}[m]
    cl(f"tab2.v.TNN.{m}", T, row, pr[0], "scalar", recipe="lowdin", args=dict(material=m, model="TNN", q="v"), display=dict(decimals=2),
       units="eV A", quantity="Dirac velocity v = a t (Lowdin, TNN GGA, active spin, SOC)", tests=["R7"])
    cl(f"tab2.kappa.TNN.{m}", T, row, pr[1], "scalar", recipe="lowdin", args=dict(material=m, model="TNN", q="kappa"),
       display=dict(decimals=2, sign="always" if pr[1][0] == "+" else "auto"), units="eV A^2", quantity="kappa TNN", tests=["R7"])
    cl(f"tab2.kappa.NN.{m}", T, row, pr[2], "scalar", recipe="lowdin", args=dict(material=m, model="NN", q="kappa"),
       display=dict(decimals=2, sign="always" if pr[2][0] == "+" else "auto"), units="eV A^2", quantity="kappa NN", tests=["R7"])
    cl(f"tab2.kappa.FP.{m}", T, row, pr[3], "literature", recipe="lit", args=dict(file="kormanyos_2015.json", path=["table6", "rows", "kappa_dn [eV A^2]", "DFT", i]),
       display=dict(decimals=2), units="eV A^2", lit=dict(ref="Kormanyos 2D Mater. 2, 022001", page=21, table="6", column="DFT", spin_label="\u2193 (band-edge A pair, p.20)"),
       tests=["L4", "T2"])
    for j, model in enumerate(("TNN", "NN")):
        cl(f"tab2.sign.{model}.{m}", T, row, "+", "sign", nth=None, sign_slot=j, recipe="edge7_sign", args=dict(material=m, model=model),
           display=dict(mode="sign"), quantity=f"sign of chi(E_g+0.01) {model} GGA",
           producer=prod("edge_v7.py", f"edge7_{m}_{model}.json", argv=[m, model, "0.0007", "2", "GGA"], extract="json:chi_SI_per_T"))
    cl(f"tab2.sign.FP.{m}", T, row, "+", "sign", nth=None, sign_slot=2, recipe="kp_fp_sign", args=dict(material=m), display=dict(mode="sign"),
       quantity="sign of the first-principles k.p threshold chi", source=dict(data="kp_out:kp_fp_all_v7.json"), producer=dict(script="kp/kp_fp_all_v7.py"))
cl("psf.thr.min", T, "changes the threshold $\\chi$ by $5", "5", "range_min", recipe="psf_thr", args=dict(agg="min"),
   display=dict(scale=0.01, decimals=0), recompute=dict(fast=dict(skip=True), full=dict(tol_rel=1e-6)),
   quantity="|chi_ED/chi_E - 1| at E_g+0.01 eV (phase-space factor added to the moment energy shift), TNN GGA, min over materials",
   notes="engine recompute on disks R=0.9/a, h=0.002/a (F at this grid equals edge7 to 2e-4); audit B5 had MoS2 +5.5%, WSe2 +21.6% only")
cl("psf.thr.max", T, "changes the threshold $\\chi$ by $5$--$40", "40", "range_max", recipe="psf_thr", args=dict(agg="max"),
   display=dict(scale=0.01, mode="ceil", step=10, decimals=0), notes="value 38.2% (WS2); printed as the bound 40%", recompute=dict(fast=dict(skip=True), full=dict(tol_rel=1e-6)), quantity="as psf.thr.min, max")
cl("fig6.v", T, "tight-binding model, $v=4.93", "4.93", "derived", recipe="lowdin", args=dict(material="MoS2", q="v", with_soc=False),
   display=dict(decimals=2), units="eV A", quantity="k.p v used for Fig. 6 (kp_params_TNN.json, no SOC in the reduction)")
cl("fig6.kappa", T, "$\\kappa=-0.18$~eV", "-0.18", "derived", recipe="lowdin", args=dict(material="MoS2", q="kappa", with_soc=False),
   display=dict(decimals=2), units="eV A^2", notes="no-SOC reduction ('from the reduction at lambda=0'); Table II uses the SOC value -0.24")
cl("fig6.w", T, "$w=-6.5$~eV", "-6.5", "derived", recipe="lowdin", args=dict(material="MoS2", q="w", with_soc=False), display=dict(decimals=1), units="eV A^2")
cl("fig6.R", T, "from disks of radius $0.6/a$ around each valley ($\\nu=10$~meV). (a)", "0.6", "method_param", value=0.6, display=dict(decimals=1),
   units="1/a", notes="kp/kp_figs_data.py R = 0.6 in the units of kp_params_TNN.json (v, kappa, beta in units of a), so 0.6/a",
   check=dict(file="kp/kp_figs_data.py", regex=r"R, NK, NU, BT = ([0-9.]+),"))
cl("fig6.nu", T, "around each valley ($\\nu=10$~meV). (a)~Magnetic", "10", "method_param", value=0.01, display=dict(scale=1e-3, decimals=0),
   units="eV", check=dict(file="kp/kp_figs_data.py", regex=r"R=R, Nk=NK, nu=([0-9.]+), want_shift=True, pairs=pairs\)\n"))
cl("fig6.B", T, "At $B=30$~T the sum is finite", "30", "method_param", value=30.0, display=dict(decimals=0), units="T",
   check=dict(file="kp/kp_figs_data.py", regex=r"BT = [0-9.]+, [0-9]+, [0-9.]+, ([0-9.]+)"))
cl("fig6.ratio3", T, "with a peak about $3", "3", "derived", recipe="kp_spectra", args=dict(what="B30_over_single"),
   display=dict(scale=0.01, decimals=0), source=dict(data="kp_out:kp_spectra.json"), quantity="max|eta K+K' 30 T| / max|eta single valley| (Fig. 6a)")
cl("fig6.shift4", T, "changes by less than $4.1", "4.1", "assertion", recipe="kp_spectra", args=dict(what="shift_change_30T_pointwise"),
   display=dict(scale=0.01, mode="less_than"), source=dict(data="kp_out:kp_spectra.json"),
   quantity="max over the plotted range of |sigma(30 T) - sigma(0)|/|sigma(0)| (Fig. 6b)",
   notes="value 4.09% at E_g-0.010 eV")
cl("fig6.shift4.B", T, "less than $4.1\\%$ at $30", "30", "method_param", value=30.0, display=dict(decimals=0), units="T")
cl("fig6.shift4.where", T, "within $10$~meV of the absorption onset", "10", "derived", recipe="kp_spectra", args=dict(what="shift_change_30T_argmax_offset"),
   display=dict(scale=1e-3, decimals=0), units="eV", source=dict(data="kp_out:kp_spectra.json"), quantity="|omega - E_g| where the shift change is largest")
cl("fig6.cut", T, "curves are cut at $E_g+0.45", "0.45", "method_param", value=0.45, display=dict(decimals=2), units="eV",
   notes="kp_spectra plotting window (fig_main.py v6 / claims recipe kp_spectra W = om <= Eg+0.45)")

# ------------------------------------------------------------------------------------------------ RESULTS: numerical method
cl("param.maps.nu", R, "calculations we take $\\nu = 10", "10", "method_param", value=0.01, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/maps_v7.py", regex=r"nu = float\(sys\.argv\[5\]\) if len\(sys\.argv\) > 5 else ([0-9.]+)"))
for n_ in (0, 1):
    cl(f"param.maps.N.{'ab'[n_]}", R, "Brillouin zone on a $600\\times600", "600", "method_param", value=600.0, display=dict(decimals=0), nth=n_,
       check=dict(file="v7jobs/maps_v7.py", regex=r"N = int\(sys\.argv\[2\]\) if len\(sys\.argv\) > 2 else ([0-9]+)"))
    cl(f"conv.N.vH.N.{'ab'[n_]}", R, "peak a $900\\times900", "900", "method_param", value=900.0, display=dict(decimals=0), nth=n_,
       source=dict(data="v7_checks:c2_N_convergence.json"), notes="c2_N_convergence.json keys chi600/chi900 (B = 1 T, nu = 10 meV)")
cl("conv.N.vH", R, "mesh changes $\\chi$ by less than $0.1", "0.1", "assertion", recipe="conv_c2", display=dict(scale=0.01, mode="less_than"),
   quantity="max |chi(N=900)/chi(N=600) - 1| at the van Hove peak (chi = eta(1 T)/1 T)", tests=["C1"])
cl("param.edge.R.methods", R, "restricted to disks of radius $0.9/a$", "0.9", "method_param", value=0.9, display=dict(decimals=1), units="1/a",
   check=dict(file="v7jobs/edge_v7.py", regex=r"kl = L\.local_kpts\(([0-9.]+), h\)"))
cl("param.edge.nu.methods", R, "$K'$, with $\\nu = 5", "5", "method_param", value=0.005, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/edge_v7.py", regex=r"vel=True\); nu = ([0-9.]+)"))
cl("param.edge.h.methods", R, "meV and spacing $0.0007/a$", "0.0007", "method_param", value=0.0007, display=dict(decimals=4), units="1/a",
   check=dict(file="v7jobs/run_v7_queue.sh", regex=r"edge_v7\.py \$m \$mo ([0-9.]+) 2 GGA"))
cl("param.thr.offset.methods", R, "spacing $0.0007/a$ at $E_g+0.01", "0.01", "method_param", value=0.01, display=dict(decimals=2), units="eV",
   check=dict(file="v7jobs/edge_v7.py", regex=r"om = Eg \+ ([0-9.]+) \+"))
cl("param.fig10.h.methods", R, "$0.001/a$ for the spectra", "0.001", "method_param", value=0.001, display=dict(decimals=3), units="1/a",
   check=dict(file="v7jobs/run_v7_queue.sh", regex=r"chiw9_v7\.py \$m \$mo F ([0-9.]+)"))
cl("param.edge01.h", R, "and $0.002/a$ at $E_g+0.1", "0.002", "method_param", value=0.002, display=dict(decimals=3), units="1/a",
   check=dict(file="v7jobs/lam_v7.py", regex=r"H, NU, R = ([0-9.]+),"))
cl("param.edge01.methods", R, "$0.002/a$ at $E_g+0.1", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV",
   check=dict(file="v7jobs/lam_v7.py", regex=r"OFF = np\.array\(\[[0-9., ]*?0\.075, ([0-9.]+),"))
cl("conv.h.offset", R, "At $E_g+0.01$~eV the spacings", "0.01", "method_param", value=0.01, display=dict(decimals=2), units="eV")
cl("conv.h.h1", R, "the spacings $0.001/a$ and", "0.001", "method_param", value=0.001, display=dict(decimals=3), units="1/a",
   source=dict(data="v7_checks:c3_h_convergence.json"), notes="c3 keys h001/h0007 (B = 30 T, R = 0.9/a, nu = 5 meV)")
cl("conv.h.h2", R, "$0.001/a$ and $0.0007/a$ give", "0.0007", "method_param", value=0.0007, display=dict(decimals=4), units="1/a")
cl("conv.h.edge", R, "give $\\chi$ within $0.001", "0.001", "assertion", recipe="conv_c3", display=dict(scale=0.01, mode="less_than"),
   quantity="max |chi(h=0.0007)/chi(h=0.001) - 1| at E_g+0.01 eV, 30 T, over the 8 GGA sets", tests=["C2"])
cl("param.maps.nom", R, "use $900$ photon energies", "900", "method_param", value=900.0, display=dict(decimals=0),
   check=dict(file="v7jobs/maps_v7.py", regex=r"NOM = int\(sys\.argv\[3\]\) if len\(sys\.argv\) > 3 else ([0-9]+)"))
cl("param.maps.omlo", R, "photon energies from $0.95", "0.95", "method_param", value=0.95, display=dict(decimals=2),
   check=dict(file="v7jobs/maps_v7.py", regex=r"om = np\.linspace\(([0-9.]+)\*Eg"))
cl("param.maps.omhi", R, "E_g$ to $1.05", "1.05", "method_param", value=1.05, display=dict(decimals=2),
   check=dict(file="v7jobs/maps_v7.py", regex=r"\*Eg, ([0-9.]+)\*Emax"))
cl("const.gs.results", R, "spin g-factor $g_s = 2", "2", "method_param", value=2.0, display=dict(decimals=0),
   check=dict(file="pub_pipeline/tb_core.py", regex=r"G_S\s*=\s*([0-9.]+)"))
cl("const.muB.results", R, "$\\mu_B = 5.788", "5.788", "scalar", recipe="const", args=dict(name="mu_B"), display=dict(scale=1e-5, decimals=3),
   units="eV/T", tests=["U2"])
cl("param.B.lo", R, "is sampled from $0$ to", "0", "method_param", value=0.0, display=dict(decimals=0), units="T")
cl("param.B.hi", R, "is sampled from $0$ to $60", "60", "method_param", value=60.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/maps_v7.py", regex=r"Bs = np\.arange\(0, ([0-9.]+) \+ 1e-9, DB\)"))
cl("param.B.step.full", R, "in steps of $1$~T in the full zone", "1", "method_param", value=1.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/maps_v7.py", regex=r"DB = float\(sys\.argv\[4\]\) if len\(sys\.argv\) > 4 else ([0-9.]+)"))
cl("param.B.step.edge", R, "zone and $2$~T near the band edge", "2", "method_param", value=2.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/run_v7_queue.sh", regex=r"edge_v7\.py \$m \$mo [0-9.]+ ([0-9.]+) GGA"))
cl("param.B.step.lda", R, "band edge ($10$~T for the LDA", "10", "method_param", value=10.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/run_v7_queue.sh", regex=r"edge_v7\.py \$m \$mo [0-9.]+ ([0-9.]+) LDA"))
# figure captions (results)
cl("fig7.lin5", R, "and stays within $5\\%$ of its linear law up to $20$~T. $N", "5", "assertion", recipe="vh7_dev20", args=dict(agg="max"),
   display=dict(scale=0.01, mode="less_than"), quantity="max over 0 < B <= 20 T and materials of |eta(B)/(s0 B) - 1| at the van Hove peak, s0 = eta(1 T)/1 T")
cl("fig7.lin5.B", R, "and stays within $5\\%$ of its linear law up to $20$~T. $N", "20", "method_param", value=20.0, display=dict(decimals=0), units="T")
cl("fig7.N", R, "$20$~T. $N=600", "600", "method_param", value=600.0, display=dict(decimals=0),
   check=dict(file="v7jobs/maps_v7.py", regex=r"N = int\(sys\.argv\[2\]\) if len\(sys\.argv\) > 2 else ([0-9]+)"))
cl("fig7.nu", R, "$N=600$, $\\nu=10$~meV.}", "10", "method_param", value=0.01, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/maps_v7.py", regex=r"nu = float\(sys\.argv\[5\]\) if len\(sys\.argv\) > 5 else ([0-9.]+)"))
cl("fig8.N", R, "(TNN, GGA, $N=600", "600", "method_param", value=600.0, display=dict(decimals=0),
   check=dict(file="v7jobs/maps_v7.py", regex=r"N = int\(sys\.argv\[2\]\) if len\(sys\.argv\) > 2 else ([0-9]+)"))
cl("fig8.nu", R, "(TNN, GGA, $N=600$, $\\nu=10", "10", "method_param", value=0.01, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/maps_v7.py", regex=r"nu = float\(sys\.argv\[5\]\) if len\(sys\.argv\) > 5 else ([0-9.]+)"))
cl("fig8.Bstep", R, "meV, $B$ in steps of $1", "1", "method_param", value=1.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/maps_v7.py", regex=r"DB = float\(sys\.argv\[4\]\) if len\(sys\.argv\) > 4 else ([0-9.]+)"))
cl("fig8.wse2.Elo", R, "between about $3.0$ and", "3.0", "assertion", recipe="maps7_banding", args=dict(material="WSe2", level=0.005, what="Emin"),
   display=dict(decimals=1), human=True, units="eV",
   quantity="lowest photon energy where the WSe2 map shows field banding |d2 eta/dB2|/max|eta| > 0.5% (other maps < 0.06%); N=600/900/1200 at 3.039 eV, 60 T in checks/c4_hotspots.json")
cl("fig8.wse2.Ehi", R, "between about $3.0$ and $3.5", "3.5", "assertion", recipe="maps7_banding", args=dict(material="WSe2", level=0.005, what="Emax"),
   display=dict(decimals=1), human=True, units="eV", quantity="highest such photon energy")
cl("fig8.wse2.B", R, "and above about $20$~T the map", "20", "assertion", recipe="maps7_banding", args=dict(material="WSe2", level=0.0075, what="Bmin"),
   display=dict(decimals=0), human=True, units="T", quantity="lowest field with banding > 0.75% (15 T at 0.5%, 23 T at 1%)")
cl("fig8.hot.MoS2.E", R, "and $\\mathrm{WS_2}$ maps near $2.4", "2.4", "assertion", human=True, value=2.4, display=dict(decimals=1), units="eV",
   quantity="MoS2 N-convergence hotspot (checks/c4_hotspots.json point 2.423 eV, 35 T; C3-violation region 2.36-2.55 eV)")
cl("fig8.hot.WS2.E", R, "near $2.4$ and $2.7", "2.7", "assertion", human=True, value=2.7, display=dict(decimals=1), units="eV",
   quantity="WS2 hotspot (c4 point 2.622 eV, 57 T; C3-violation > 0.05 region 2.60-2.79 eV, centre 2.70)")
for n_ in (0, 1):
    cl(f"fig8.hot.N.{'ab'[n_]}", R, "eV a $900\\times900", "900", "method_param", value=900.0, display=dict(decimals=0), nth=n_,
       source=dict(data="v7_checks:c4_hotspots.json"), notes="c4_hotspots.json N = 600/900/1200")
cl("fig8.hot.max", R, "changes $\\eta^{x;xx}$ by up to $7", "7", "derived", recipe="c4_change", args=dict(materials=["MoS2", "WS2"]),
   display=dict(scale=0.01, decimals=0), quantity="max over the MoS2/WS2 hotspots of |eta(N=900)/eta(N=600) - 1| (F model)")
cl("fig9.offset", R, "$\\hbar\\omega = E_g + 0.01$~eV for (a)~$\\mathrm{MoS_2}$, (b)~$\\mathrm{MoSe_2}$, (c)~$\\mathrm{WS_2}$, (d)~$\\mathrm{WSe_2}$ (TNN, GGA; local grid of spacing",
   "0.01", "method_param", value=0.01, display=dict(decimals=2), units="eV", check=dict(file="v7jobs/edge_v7.py", regex=r"om = Eg \+ ([0-9.]+) \+"))
cl("fig9.h", R, "(TNN, GGA; local grid of spacing $0.0007/a$", "0.0007", "method_param", value=0.0007, display=dict(decimals=4), units="1/a",
   check=dict(file="v7jobs/run_v7_queue.sh", regex=r"edge_v7\.py \$m \$mo ([0-9.]+) 2 GGA"))
cl("fig9.R", R, "$0.0007/a$ over disks of radius $0.9/a$", "0.9", "method_param", value=0.9, display=dict(decimals=1), units="1/a",
   check=dict(file="v7jobs/edge_v7.py", regex=r"kl = L\.local_kpts\(([0-9.]+), h\)"))
cl("fig9.nu", R, "radius $0.9/a$, $\\nu=5$~meV, averaged over $5$~meV of photon", "5", "method_param", value=0.005, display=dict(scale=1e-3, decimals=0),
   units="eV", check=dict(file="v7jobs/edge_v7.py", regex=r"vel=True\); nu = ([0-9.]+)"))
cl("fig9.avg", R, "averaged over $5$~meV of photon energy", "5", "derived", recipe="edge7_omega_window", display=dict(scale=1e-3, decimals=0), units="eV",
   quantity="span of the 5 photon energies averaged in the edge runs (E_g+0.0075 to E_g+0.0125 eV)")
cl("fig10.B", R, "$\\chi=\\eta^{x;xx}(B)/B$ at $B=30", "30", "method_param", value=30.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/chiw9_v7.py", regex=r"B = ([0-9.]+); r = L\.eta"))
cl("fig10.h", R, "(GGA, local grid of spacing $0.001/a$", "0.001", "method_param", value=0.001, display=dict(decimals=3), units="1/a",
   check=dict(file="v7jobs/run_v7_queue.sh", regex=r"chiw9_v7\.py \$m \$mo F ([0-9.]+)"))
cl("fig10.R", R, "$0.001/a$ over disks of radius $0.9/a$ around", "0.9", "method_param", value=0.9, display=dict(decimals=1), units="1/a",
   check=dict(file="v7jobs/chiw9_v7.py", regex=r"local_kpts\(([0-9.]+), h\)"))
cl("fig10.nu", R, "$K'$, $\\nu=5$~meV, averaged", "5", "method_param", value=0.005, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/chiw9_v7.py", regex=r"om, ([0-9.]+), FL, kpts"))
cl("fig10.avg", R, "averaged over $5$~meV) for", "5", "method_param", value=0.005, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/fig_v7_all.py", regex=r"np\.convolve\(raw, np\.ones\(([0-9]+)\)/[0-9]+, \"same\"\)", scale=0.001),
   notes="5-point running mean on the 1 meV offset grid of chiw9_v7.py (fig_v7_all.py fig10)")
cl("fig10.lin.TNN", R, "by less than $0.2\\%$ near threshold in the TNN", "0.2", "assertion", recipe="edge7_lin30", args=dict(model="TNN"),
   display=dict(scale=0.01, mode="less_than"), quantity="max |eta(30 T)/30 T - chi|/|chi| over the TNN edge runs (chi = B = 0 slope s)",
   notes="the caption compares with the zero-field slope; against d eta/dB at 30 T the difference is 2x larger (0.29%)")
cl("fig10.lin.NN", R, "TNN model and about $1.5\\%$ in the NN model", "1.5", "derived", recipe="edge7_lin30", args=dict(model="NN"),
   display=dict(scale=0.01, decimals=1), quantity="same for the NN edge runs",
   notes="edge7 data: 1.509% (NN MoS2); the s B + c B^3 fit alone gives 1.498%; vs d eta/dB at 30 T 3.0%")
cl("fig11.offset", R, "(a)~$\\chi$ at $\\hbar\\omega=E_g+0.01", "0.01", "method_param", value=0.01, display=dict(decimals=2), units="eV",
   check=dict(file="v7jobs/edge_v7.py", regex=r"om = Eg \+ ([0-9.]+) \+"))
cl("fig11.nu", R, "(Table~\\ref{tab:kappa}). $\\nu=5", "5", "method_param", value=0.005, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/edge_v7.py", regex=r"vel=True\); nu = ([0-9.]+)"))
cl("fig11.r", R, "GGA tight-binding points ($r=-0.98", "-0.98", "fit_coef", recipe="fit11", args=dict(what="r"), display=dict(decimals=2),
   quantity="correlation of the 8 GGA TB threshold chi with the Lowdin kappa")
# van Hove section
cl("vH.E.B", R, "where $|\\eta^{x;xx}|$ at $60$~T is largest", "60", "method_param", value=60.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/maps_v7.py", regex=r"Bs = np\.arange\(0, ([0-9.]+) \+ 1e-9, DB\)"),
   notes="largest field of prod7; decomp_v7.py takes vH = argmax|eta[-1]| of the finished prod7 map")
for m, e in zip(MATS, ("3.35", "3.05", "3.60", "2.99")):
    cl(f"vH.E.{m}", R, "is largest ($3.35$, $3.05$, $3.60$, and $2.99$~eV", e, "scalar", recipe="vh7_energy", args=dict(material=m),
       display=dict(decimals=2), units="eV", quantity=f"van Hove photon energy argmax|eta(60 T)| {m} (prod7)", producer=prod("maps_v7.py", f"prod7_{m}_F.npz"),
       tests=["S4"])
for m, c in zip(MATS, ("+2.9", "+5.3", "+4.7", "+6.5")):
    cl(f"vH.chi.{m}", R, "small-field slope $\\chi$ is $+2.9$, $+5.3$, $+4.7$, and $+6.5", c, "scalar", recipe="vh7_slope",
       args=dict(material=m), display=dict(scale=1e-2, decimals=1, sign="always"), units=SI_T,
       quantity=f"van Hove chi {m} = eta(1 T)/1 T at argmax|eta(60 T)| (prod7)", producer=prod("maps_v7.py", f"prod7_{m}_F.npz"), tests=["S2", "C1"],
       notes="c2_N_convergence: N=900 changes it by < 0.05%; decomp7 (+-10 T odd part) agrees to 1%")
cl("vH.eta60.min", R, "reaches $1.6", "1.6", "range_min", recipe="vh7_eta60", args=dict(agg="min"), display=dict(decimals=1), units="A m V^-2 s^-1")
cl("vH.eta60.max", R, "reaches $1.6$--$3.1", "3.1", "range_max", recipe="vh7_eta60", args=dict(agg="max"), display=dict(decimals=1), units="A m V^-2 s^-1")
cl("vH.eta60.B", R, "at $B = \\SI{60}{T}$. With", "60", "method_param", value=60.0, display=dict(decimals=0), units="T")
cl("vH.FZ.min", R, "smaller by a factor of $1.5", "1.5", "range_min", recipe="decomp7_ratio_range", args=dict(where="vH", num="F", den="Z", agg="min"),
   display=dict(decimals=1), quantity="chi_F/chi_Z at the van Hove peak (decomp7, +-10 T odd part, N=600)", producer=prod("decomp_v7.py", "decomp7_{m}.json"))
cl("vH.FZ.max", R, "smaller by a factor of $1.5$ to $2.2", "2.2", "range_max", recipe="decomp7_ratio_range", args=dict(where="vH", num="F", den="Z", agg="max"),
   display=dict(decimals=1), producer=prod("decomp_v7.py", "decomp7_{m}.json"))
cl("vH.lin5", R, "The response stays within $5", "5", "assertion", recipe="vh7_dev20", args=dict(agg="max"), display=dict(scale=0.01, mode="less_than"),
   quantity="as fig7.lin5")
cl("vH.lin5.B", R, "stays within $5\\%$ of its linear law up to $20$~T, which", "20", "method_param", value=20.0, display=dict(decimals=0), units="T")
cl("vH.dev60.B", R, "At $60$~T it falls", "60", "method_param", value=60.0, display=dict(decimals=0), units="T")
cl("vH.dev60.min", R, "extrapolation by $3", "3", "range_min", recipe="vh7_below60", args=dict(agg="min"), display=dict(scale=0.01, decimals=0),
   quantity="1 - eta(60 T)/(s0 60 T) at the van Hove peak, min over materials")
cl("vH.dev60.max", R, "extrapolation by $3$--$29", "29", "range_max", recipe="vh7_below60", args=dict(agg="max"), display=dict(scale=0.01, decimals=0),
   notes="WSe2 28.9% at N=600; c4_hotspots: eta(60 T) changes by 1.4% to N=1200 (27.9%)")
cl("maps.weight.lo", R, "lies between about $2.2", "2.2", "assertion", human=True, value=2.2, display=dict(decimals=1), units="eV",
   quantity="lower edge of the main spectral weight (Fig. 8); recipe maps7_weight_fraction(2.2, 4.4) = 0.887 (MoSe2) .. 0.986 inside")
cl("maps.weight.hi", R, "between about $2.2$ and $4.4", "4.4", "assertion", human=True, value=4.4, display=dict(decimals=1), units="eV")
# band edge section
cl("edge.R.results", R, "comes from within $0.9/a$", "0.9", "method_param", value=0.9, display=dict(decimals=1), units="1/a",
   check=dict(file="v7jobs/edge_v7.py", regex=r"kl = L\.local_kpts\(([0-9.]+), h\)"),
   notes="the grid radius of the edge runs; no v7 R-convergence file (earlier checks)")
cl("param.thr.offset.results", R, "plots $\\eta^{x;xx}(B)$ at $\\hbar\\omega=E_g+0.01", "0.01", "method_param", value=0.01, display=dict(decimals=2), units="eV",
   check=dict(file="v7jobs/edge_v7.py", regex=r"om = Eg \+ ([0-9.]+) \+"))
for model, near, prs in (("TNN", "The TNN model gives $\\chi=+1.9$, $+0.2$, $+0.06$, and $-0.5", ("+1.9", "+0.2", "+0.06", "-0.5")),
                         ("NN", "the NN model gives $+0.2$, $-0.9$, $+4.7$, and $+3.1", ("+0.2", "-0.9", "+4.7", "+3.1"))):
    for m, pr in zip(MATS, prs):
        cl(f"chi.thr.{model}.GGA.{m}", R, near, pr, "scalar", recipe="edge7_chi", args=dict(material=m, model=model),
           display=dict(scale=1e-4, decimals=len(pr.split('.')[1]), sign="always" if pr[0] == "+" else "auto"),
           units=SI_T, quantity=f"threshold chi = edge7 chi_SI_per_T at E_g+0.01 eV, {model} GGA F",
           producer=prod("edge_v7.py", f"edge7_{m}_{model}.json", argv=[m, model, "0.0007", "2", "GGA"], extract="json:chi_SI_per_T",
                         params=dict(R=0.9, h=0.0007, nu=0.005)), tests=EDGE_TESTS)
cl("edge.lin60", R, "linear law by less than $1", "1", "assertion", recipe="edge7_dev60", args=dict(model="TNN"),
   display=dict(scale=0.01, mode="less_than"), quantity="max |c 60^2/s| over the TNN GGA edge runs")
cl("edge.lin60.B", R, "less than $1\\%$ at $60", "60", "method_param", value=60.0, display=dict(decimals=0), units="T")
cl("signchange.count", R, "In three cases with small", "three", "count", recipe="chiw7_early_count", display=dict(mode="word"),
   quantity="number of the 8 GGA chi(omega) series (5 meV mean) whose first sign change lies below E_g+0.1 eV")
for key, near, pr in (("TNN.MoSe2", "at $E_g+0.07", "0.07"), ("TNN.WS2", "and $E_g+0.03", "0.03"), ("NN.MoS2", "and at $E_g+0.02", "0.02")):
    mo, m = key.split(".")
    cl(f"signchange.{key}", R, near, pr, "scalar", recipe="chiw7_signchange", args=dict(material=m, model=mo), display=dict(decimals=2), units="eV",
       quantity="first sign change of chi(omega) (5 meV running mean) above E_g", producer=prod("chiw9_v7.py", f"chiwR9v7_{m}_{mo}_F_0.001.json"))
cl("signchange.persist", R, "persists to $E_g+0.16", "0.16", "assertion", recipe="chiw7_persist", display=dict(mode="floor", step=0.01, decimals=2),
   units="eV", quantity="min over the other five series of the energy to which the threshold sign persists (window end 0.45 eV if never)")
cl("fit.count", R, "All eight tight-binding points", "eight", "count", recipe="fit11", args=dict(what="count_opposite"), display=dict(mode="word"),
   quantity="GGA TB points with sign(chi) = -sign(kappa)")
cl("fit.slope", R, "follow $\\chi\\approx(-9.8", "-9.8", "fit_coef", recipe="fit11", args=dict(what="slope"), display=dict(decimals=1),
   units="1e-4 A m V^-2 s^-1 T^-1 per eV A^2", quantity="slope of the linear fit of the 8 GGA TB chi (1e-4 SI) vs Lowdin kappa")
cl("fit.icpt", R, "\\kappa-0.1)\\times10^{-4}", "-0.1", "fit_coef", recipe="fit11", args=dict(what="icpt"), display=dict(decimals=1), units="1e-4 A m V^-2 s^-1 T^-1")
cl("fit.r", R, "(correlation $r=-0.98", "-0.98", "fit_coef", recipe="fit11", args=dict(what="r"), display=dict(decimals=2))
cl("kfp.range.lo", R, "First-principles fits give $\\kappa=-0.9", "-0.9", "range_max", recipe="lit_range",
   args=dict(file="kormanyos_2015.json", path=["table6", "rows", "kappa_dn [eV A^2]", "DFT"], agg="max"), display=dict(decimals=1), units="eV A^2", tests=["L4"])
cl("kfp.range.hi", R, "$\\kappa=-0.9$ to $-1.5", "-1.5", "range_min", recipe="lit_range",
   args=dict(file="kormanyos_2015.json", path=["table6", "rows", "kappa_dn [eV A^2]", "DFT"], agg="min"), display=dict(decimals=1, rounding="half_away"),
   units="eV A^2", notes="tie: the table value -1.45 (MoS2) prints -1.5 rounding half away from zero (-1.4 half to even)")
cl("kfp.over.tnn", R, "at least six times the magnitude", "six", "assertion", recipe="kfp_over_tnn_min", display=dict(mode="at_least_word"),
   quantity="min over materials of |kappa_FP (kappa_dn)|/|kappa_TNN|")
cl("tb.kappa.fit16", R, "the TNN bands give about $-0.7", "-0.7", "derived", recipe="tb_warp", args=dict(q="kappa_fit16", agg="mean"),
   display=dict(decimals=1), units="eV A^2",
   quantity="Kormanyos-style fit of the TNN bands over 16% of |Gamma K| (no diagonal cubic term), c3/(2|v|), mean of the four",
   notes="per material -0.75, -0.69, -0.69, -0.70 (audit X2 tb_warping.py: same)")
cl("tb.kappa.disp.lo", R, "coefficient gives $-0.8", "-0.8", "range_max", recipe="tb_warp", args=dict(q="kappa_disp", agg="max"), display=dict(decimals=1),
   units="eV A^2", quantity="exact q^3 cos3phi coefficient of d^2 of the TNN bands, c3/(2|v|), max (least negative)")
cl("tb.kappa.disp.hi", R, "$-0.8$ to $-1.0", "-1.0", "range_min", recipe="tb_warp", args=dict(q="kappa_disp", agg="min"), display=dict(decimals=1),
   units="eV A^2")
cl("param.thr.offset.kp", R, "gives $\\chi(E_g+0.01", "0.01", "method_param", value=0.01, display=dict(decimals=2), units="eV",
   check=dict(file="kp/kp_fp_all_v7.py", regex=r"OFF = np\.array\(\[([0-9.]+),"))
cl("kp.fp.chi.min", R, "\\mathrm{eV})=+0.7", "+0.7", "range_min", recipe="kp_fp_chi", args=dict(agg="min"),
   display=dict(scale=1e-3, decimals=1, sign="always"), units=SI_T, source=dict(data="kp_out:kp_fp_all_v7.json"),
   producer=dict(script="kp/kp_fp_all_v7.py"), tests=["S6"])
cl("kp.fp.chi.max", R, "\\mathrm{eV})=+0.7$ to $+1.5", "+1.5", "range_max", recipe="kp_fp_chi", args=dict(agg="max"),
   display=dict(scale=1e-3, decimals=1, sign="always"), units=SI_T, source=dict(data="kp_out:kp_fp_all_v7.json"), producer=dict(script="kp/kp_fp_all_v7.py"))
cl("fit.atFP.min", R, "first-principles $\\kappa$ gives $+0.9", "+0.9", "derived", recipe="fit11", args=dict(what="atFP_min"),
   display=dict(scale=1e-3, decimals=1, sign="always"), units=SI_T, quantity="fit (fit.slope, fit.icpt) evaluated at the four kappa_FP (kappa_dn)")
cl("fit.atFP.max", R, "first-principles $\\kappa$ gives $+0.9$ to $+1.4", "+1.4", "derived", recipe="fit11", args=dict(what="atFP_max"),
   display=dict(scale=1e-3, decimals=1, sign="always"), units=SI_T)
cl("lda.count", R, "(open symbols). All eight have", "eight", "count", recipe="lda_count_opposite", display=dict(mode="word"),
   quantity="LDA parameter sets (TNN and NN) with sign(chi) = -sign(kappa_LDA)",
   producer=prod("edge_v7.py", "edge7_{m}_{model}_LDA.json", argv=["{m}", "{model}", "0.0007", "10", "LDA"]))
cl("lda.kappa.MoS2", R, "where LDA gives $\\kappa=+0.01", "+0.01", "scalar", recipe="lowdin", args=dict(material="MoS2", model="TNN", functional="LDA"),
   display=dict(decimals=2, sign="always"), units="eV A^2", tests=["R7"])
cl("lda.chi.MoS2", R, "\\AA$^2$ and $\\chi=-0.1", "-0.1", "scalar", recipe="edge7_chi", args=dict(material="MoS2", model="TNN", functional="LDA"),
   display=dict(scale=1e-4, decimals=1), units=SI_T,
   quantity="threshold chi, TNN LDA MoS2 (must be < 0: 'opposite to kappa' and 'opposite to its GGA sign')",
   producer=prod("edge_v7.py", "edge7_MoS2_TNN_LDA.json", argv=["MoS2", "TNN", "0.0007", "10", "LDA"], extract="json:chi_SI_per_T"),
   notes="final edge7_MoS2_TNN_LDA.json (7 fields, 10 T steps) from the Mac")
cl("kp.fp.order.results", R, "threshold field coefficient of about $10^{-3}", "10", "assertion", recipe="kp_fp_order", display=dict(mode="order", exponent=-3))
cl("param.edge01.lambda", R, "near-edge response ($E_g+0.1", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV",
   check=dict(file="v7jobs/lam_v7.py", regex=r"OFF = np\.array\(\[[0-9., ]*?0\.075, ([0-9.]+),"))

# ------------------------------------------------------------------------------------------------ DISCUSSION
cl("edge.lin60.disc", D, "TNN response stays linear within $1", "1", "assertion", recipe="edge7_dev60", args=dict(model="TNN"),
   display=dict(scale=0.01, mode="less_than"), quantity="as edge.lin60")
cl("edge.lin60.disc.B", D, "within $1\\%$ up to $60", "60", "method_param", value=60.0, display=dict(decimals=0), units="T")
cl("count.param.sets", D, "In all sixteen tight-binding", "sixteen", "count", recipe="all16_opposite", display=dict(mode="word"),
   quantity="TB parameter sets {NN,TNN} x {GGA,LDA} x 4 materials with sign(chi) = -sign(kappa) at threshold")
cl("tnn.kappa.range.lo", D, "between $-0.24", "-0.24", "range_min", recipe="lowdin_range", args=dict(model="TNN", agg="min"), display=dict(decimals=2), units="eV A^2")
cl("tnn.kappa.range.hi", D, "between $-0.24$ and $+0.08", "+0.08", "range_max", recipe="lowdin_range", args=dict(model="TNN", agg="max"),
   display=dict(decimals=2, sign="always"), units="eV A^2")
cl("kp.fp.order.disc", D, "positive threshold $\\chi$ of about $10^{-3}", "10", "assertion", recipe="kp_fp_order", display=dict(mode="order", exponent=-3))
cl("est.chi", D, "With $\\chi=10^{-3}", "10", "method_param", value=-3.0, display=dict(mode="order", exponent=-3), units=SI_T,
   quantity="input of the current estimate (= current_nA default chi)")
cl("est.B", D, "near threshold, $B=30", "30", "method_param", value=30.0, display=dict(decimals=0), units="T", quantity="input of the current estimate")
cl("est.tau", D, "$\\tau_r=1", "1", "method_param", value=1.0, display=dict(decimals=0), units="ps", quantity="input of the current estimate")
cl("est.P", D, "and $1$~mW focused", "1", "method_param", value=1.0, display=dict(decimals=0), units="mW", quantity="input of the current estimate")
cl("est.area", D, "focused to $1\\,\\mu\\mathrm{m}^2", "1", "method_param", value=1.0, display=dict(decimals=0), units="um^2", quantity="input of the current estimate")
cl("est.width", D, "across a $1\\,\\mu$m channel", "1", "method_param", value=1.0, display=dict(decimals=0), units="um", quantity="input of the current estimate")
cl("current.nA", D, "this is of order $10", "10", "assertion", recipe="current_nA", display=dict(mode="order_of", lo=3, hi=30), units="nA",
   quantity="tau chi B |E(w)|^2 x 1 um, |E(w)|^2 = I/(2 c eps0) (Sipe convention; 4x larger with E0^2)")
cl("pulsed.B100", D, "pulsed fields up to $\\sim100", "100", "method_param", value=100.0, display=dict(decimals=0), units="T",
   quantity="pulsed-field magneto-optics range (Stier 2016, Goryca 2019)", human=True)
cl("dep.B60", D, "A departure below $60", "60", "method_param", value=60.0, display=dict(decimals=0), units="T", quantity="largest computed field")
cl("pert.B100", D, "controlled only up to about $100", "100", "method_param", value=100.0, display=dict(decimals=0), units="T")
cl("valley.g.lo", D, "valley $g$-factor of $-1.2", "-1.2", "range_max", recipe="valley_g_range", args=dict(agg="max"), display=dict(decimals=1),
   quantity="A-transition valley g with H_Z + itinerant moment (TNN GGA)")
cl("valley.g.hi", D, "of $-1.2$ to $-1.9", "-1.9", "range_min", recipe="valley_g_range", args=dict(agg="min"), display=dict(decimals=1))
cl("lit.gvalley.exp", D, "against $\\approx-4", "4", "literature", value=4.0, display=dict(decimals=0), human=True,
   lit=dict(ref="Srivastava 2015, MacNeill 2015, Stier 2016", quantity="measured A-exciton valley g-factor magnitude (about -4)"))
cl("lit.mote2.1T", D, "lies only about $43", "43", "literature", value=0.043, display=dict(scale=1e-3, decimals=0), units="eV", human=True,
   lit=dict(ref="Duerloo, Li, Reed, Nat. Commun. 5, 4214 (2014)", quote="the 1T'-2H energy offset thus calculated for MoTe2 is sufficiently small (43 meV)",
            note="PBE, per formula unit (stage-2 reviewer, S20)"))

# ------------------------------------------------------------------------------------------------ APPENDIX
cl("const.muB.app", A, "Bohr magneton ($\\SI{5.788e-5}", "5.788e-5", "scalar", recipe="const", args=dict(name="mu_B"), display=dict(mode="sci", decimals=3), units="eV/T")
cl("const.gs.app", A, "applied field, $g_s = 2", "2", "method_param", value=2.0, display=dict(decimals=0),
   check=dict(file="pub_pipeline/tb_core.py", regex=r"G_S\s*=\s*([0-9.]+)"))
cl("ka.range.lo", A, "\\approx 1.3", "1.3", "range_min", recipe="ka_range", args=dict(agg="min"), display=dict(decimals=1), units="1/eV", tests=["U1"])
cl("ka.range.hi", A, "\\approx 1.3$--$1.5", "1.5", "range_max", recipe="ka_range", args=dict(agg="max"), display=dict(decimals=1), units="1/eV")
cl("D.max60.app", A, "stays below $3", "3", "assertion", recipe="D_max", args=dict(B=60.0), display=dict(scale=0.01, mode="less_than"))
cl("D.max60.app.B", A, "below $3\\%$ at $\\SI{60}", "60", "method_param", value=60.0, display=dict(decimals=0), units="T")
for n_, near in enumerate(["\\eqref{eq:app_ring} at $v=1", "\\rangle_\\phi$ ($v=1"]):
    cl(f"tab4.v.{n_}", A, near, "1", "method_param", value=1.0, display=dict(decimals=0), nth=0)
    cl(f"tab4.m.{n_}", A, near, "0.83", "method_param", value=0.83, display=dict(decimals=2))
    cl(f"tab4.kappa.{n_}", A, near, "0.25", "method_param", value=0.25, display=dict(decimals=2))
cl("tab4.sigfigs", A, "reproduced to five significant", "five", "assertion", recipe="ring_shift_sigfigs", display=dict(mode="at_least_word"))
for k, near, ex, cf, sc in ((0.03, "$0.03$ & injection", "3.922", "3.935", 1e-4), (0.05, "$0.05$ & injection", "1.083", "1.093", 1e-3),
                            (0.08, "$0.08$ & injection", "2.733", "2.798", 1e-3)):
    cl(f"tab4.k.{k}", A, near, f"{k:.2f}", "method_param", value=k, display=dict(decimals=2))
    cl(f"tab4.inj.exact.{k}", A, near, ex, "scalar", recipe="ring", args=dict(k=k, channel="inj_exact"), display=dict(scale=sc, decimals=3))
    cl(f"tab4.inj.closed.{k}", A, near, cf, "scalar", recipe="ring", args=dict(k=k, channel="inj_closed"), display=dict(scale=sc, decimals=3))
cl("tab4.shift.exact", A, "$\\to0$ & shift", "1.8145", "scalar", recipe="ring", args=dict(k=1e-4, channel="shift_exact"), display=dict(scale=1e-1, decimals=4), nth=0)
cl("tab4.shift.closed", A, "$\\to0$ & shift", "1.8145", "scalar", recipe="ring", args=dict(k=0.0, channel="shift_closed"), display=dict(scale=1e-1, decimals=4), nth=1)
cl("tab5.offset", A, "at $\\hbar\\omega = E_g + 0.1$~eV, from disks", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV",
   check=dict(file="kp/kp_figs_data.py", regex=r"o = np\.array\(\[Eg\+([0-9.]+)\]\)"))
cl("tab5.R", A, "from disks of radius $0.6/a$ around each valley with", "0.6", "method_param", value=0.6, display=dict(decimals=1), units="1/a",
   check=dict(file="kp/kp_figs_data.py", regex=r"R, NK, NU, BT = ([0-9.]+),"), notes="as fig6.R (parameters in units of a)")
cl("tab5.nu", A, "around each valley with $\\nu = 10", "10", "method_param", value=0.01, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="kp/kp_figs_data.py", regex=r"o, p=base, R=R, Nk=NK, nu=([0-9.]+), want_shift=True"))
cl("tab5.B.cap", A, "unchanged by $30", "30", "method_param", value=30.0, display=dict(decimals=0), units="T")
cl("tab5.B.head", A, "$K\\!+\\!K'$, $B=30", "30", "method_param", value=30.0, display=dict(decimals=0), units="T")
for key, near, pr, sc, entry in (("lpge.B30", "$\\approx0$ & $+6.50", "+6.50", 1e-3, "KKp_B30:lpge_xxx"),
                                 ("lpge.K", "\\times10^{-3}$ & $+1.18", "+1.18", 1e-1, "K_B0:lpge_xxx")):
    cl(f"tab5.{key}", A, near, pr, "scalar", recipe="kp_table", args=dict(entry=entry), display=dict(scale=sc, decimals=2, sign="always"),
       units="A m V^-2 s^-1", source=dict(data="kp_out:kp_table.json", extract=f"recipe:kp_table_data:{entry}"),
       producer=dict(script="kp/kp_figs_data.py", note="cloud paths /home/claude/pub; inputs kp_params_TNN.json"),
       recompute=dict(fast=dict(args=dict(Nk=101), tol_vs_value=0.05), full=dict(args=dict(Nk=401), tol_rel=1e-6)))
for nth, (key, entry) in enumerate((("shift.B0", "KKp_B0:shift_yyy"), ("shift.B30", "KKp_B30:shift_yyy"), ("shift.K", "K_B0:shift_yyy"))):
    cl(f"tab5.{key}", A, "($\\mathcal{T}$-even)     & $0.795", ("0.795", "0.795", "0.397")[nth], "scalar", nth=(0, 1, 0)[nth],
       recipe="kp_table", args=dict(entry=entry), display=dict(decimals=3), units="nm muA V^-2",
       source=dict(data="kp_out:kp_table.json", extract=f"recipe:kp_table_data:{entry}"), producer=dict(script="kp/kp_figs_data.py"),
       recompute=dict(fast=dict(args=dict(Nk=101), tol_vs_value=0.05), full=dict(args=dict(Nk=401), tol_rel=1e-6)))
cl("fig12.offset", A, "$\\hbar\\omega = E_g + 0.1$~eV from the TNN", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV",
   check=dict(file="v7jobs/lam_v7.py", regex=r"OFF = np\.array\(\[[0-9., ]*?0\.075, ([0-9.]+),"))
cl("fig12.B", A, "at $B=\\pm10", "10", "method_param", value=10.0, display=dict(decimals=0), units="T",
   check=dict(file="v7jobs/lam_v7.py", regex=r"None, ([0-9.]+), om, NU, FLAGS, kpts=kp, lam=lam\)"))
cl("fig12.h", A, "(local grid of spacing $0.002/a$", "0.002", "method_param", value=0.002, display=dict(decimals=3), units="1/a",
   check=dict(file="v7jobs/lam_v7.py", regex=r"H, NU, R = ([0-9.]+),"))
cl("fig12.R", A, "$0.002/a$ over disks of radius $0.9/a$, $\\nu=5$~meV).}", "0.9", "method_param", value=0.9, display=dict(decimals=1), units="1/a",
   check=dict(file="v7jobs/lam_v7.py", regex=r"H, NU, R = [0-9.]+, [0-9.]+, ([0-9.]+)"))
cl("fig12.nu", A, "radius $0.9/a$, $\\nu=5$~meV).}", "5", "method_param", value=0.005, display=dict(scale=1e-3, decimals=0), units="eV",
   check=dict(file="v7jobs/lam_v7.py", regex=r"H, NU, R = [0-9.]+, ([0-9.]+),"))
cl("kp.lam.lo", A, "is varied from $0$ to $0.25", "0", "method_param", value=0.0, display=dict(decimals=0), units="eV")
cl("kp.lam.hi", A, "is varied from $0$ to $0.25", "0.25", "method_param", value=0.25, display=dict(decimals=2), units="eV",
   notes="recipe kp_lam_change sweeps lambda = 0, 0.05, ..., 0.25 eV")
cl("kp.lam.offset01", A, "fixed, $\\chi$ at $E_g+0.1", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV")
cl("kp.lam.change01", A, "changes by at most $3.2", "3.2", "assertion", recipe="kp_lam_change", args=dict(offset=0.1),
   display=dict(scale=0.01, mode="ceil", step=0.1, decimals=1), recompute=dict(fast=dict(skip=True), full=dict(tol_rel=1e-6)),
   quantity="max over materials and lambda in [0, 0.25] eV of |chi(lambda)/chi(0) - 1|, two-band, TNN-derived params, E_g+0.1 eV, 30 T, R=0.6/a, Nk=401, nu=5 meV",
   notes="kp2 recompute (~1 min); same as audit t5/lam_kp_tnn.py")
cl("kp.lam.offset001", A, "and at $E_g+0.01$~eV by at most", "0.01", "method_param", value=0.01, display=dict(decimals=2), units="eV")
cl("kp.lam.change001", A, "by at most $8", "8", "assertion", recipe="kp_lam_change", args=dict(offset=0.01),
   display=dict(scale=0.01, mode="ceil", step=1, decimals=0), recompute=dict(fast=dict(skip=True), full=dict(tol_rel=1e-6)),
   quantity="as kp.lam.change01 at E_g+0.01 eV")
cl("lam.offset", A, "$\\chi$ at $E_g+0.1$~eV increases", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV",
   check=dict(file="v7jobs/lam_v7.py", regex=r"OFF = np\.array\(\[[0-9., ]*?0\.075, ([0-9.]+),"))
cl("lam.R2", A, "R^2_{\\mathrm{lin}}>0.99", "0.99", "assertion", recipe="lam7_R2", display=dict(mode="greater_than"),
   quantity="min over WS2, WSe2 of R^2 of a linear fit of chi(E_g+0.1 eV) vs lambda over [0.5, 1.5] lambda0 (lam7)",
   producer=prod("lam_v7.py", "lam7_{m}_F.npz"))
cl("ws2.kappa.lam0", A, "spin goes from $+0.13", "+0.13", "scalar", recipe="lowdin", args=dict(material="WS2", lam=0.0), display=dict(decimals=2, sign="always"), units="eV A^2")
cl("ws2.kappa.lam1", A, "at $\\lambda=0$ to $-0.02", "-0.02", "scalar", recipe="lowdin", args=dict(material="WS2"), display=dict(decimals=2), units="eV A^2")
cl("ws2.kappa.lam15", A, "$\\lambda_0$ and $-0.11", "-0.11", "scalar", recipe="lowdin", args=dict(material="WS2", lam_scale=1.5), display=dict(decimals=2), units="eV A^2")
cl("ws2.lam15", A, "at $1.5\\lambda_0", "1.5", "method_param", value=1.5, display=dict(decimals=1))
cl("mos2.lam.below", A, "Below $\\lambda\\approx0.05", "0.05", "derived", recipe="half", args=dict(x=0.1), display=dict(decimals=2), units="eV",
   quantity="lambda at which the splitting 2 lambda equals the 0.1 eV offset")
cl("mos2.split.offset", A, "drops below $0.1", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV")
cl("mos2.absorb.offset", A, "to absorb at $E_g+0.1", "0.1", "method_param", value=0.1, display=dict(decimals=1), units="eV")


# ================================================================================================ build
def fmt(value, d):
    """Printed form of a claim value under its display rule (protocol G2; round half to even)."""
    from claims_check import fmt as _f
    return _f(value, d)


def locate(text, near, printed, nth, sign_slot=None):
    i = text.find(near)
    if i < 0 or text.find(near, i + 1) >= 0:
        raise ValueError(f"anchor not unique/found: {near!r}")
    if sign_slot is not None:                # nth '+'/'-' symbol in "$+$ / $-$ / $+$" after near
        j = i + len(near)
        cnt = -1
        while True:
            mm = re.compile(r"\$([+-])\$").search(text, j)
            cnt += 1
            if cnt == sign_slot + 0:
                return mm.start(1), mm.group(1)
            j = mm.end()
    j = i
    for _ in range((nth or 0) + 1):
        j = text.find(printed, j if _ == 0 else j + 1)
        if j < 0:
            raise ValueError(f"printed {printed!r} not found after {near!r}")
    return j, printed


def anchor_for(text, pos, L):
    w = 30
    while True:
        a, b = max(0, pos - w), min(len(text), pos + L + w)
        an = text[a:b]
        if text.count(an) == 1 or w > 400:
            return an, pos - a
        w += 20


REPLACED_BY = {"fig7.linear20": "fig7.lin5", "vH.linear20": "vH.lin5", "fig10.lin02": "fig10.lin.TNN", "param.B.step": "param.B.step.full",
               "kp.lam.change": "kp.lam.change01", "fig6.shift14": "fig6.shift4", "fig6.shift14.B": "fig6.shift4.B",
               "conv.h.edge.h": "conv.h.h2", "fig9.nu": "fig9.nu", "fig.zeeman80.2": "fig.zeeman80.2"}
REASON = "final text (stage-2 source: Overleaf + 82 audit edits + 29 v7 edits) and v7 data, 2026-10-14"


def _sources(ctx, sp, e):
    """Data files behind a claim: the declared source plus every v7 file the recipe opened (sha of each)."""
    src = dict(sp.get("source", {}))
    files = src.get("data", [])
    files = [files] if isinstance(files, str) else list(files)
    files += sorted(ctx.memo.get("_access", set()) - set(files))
    if not files:
        return src or None
    shas = {}
    for f in files:
        key, rel_ = f.split(":", 1)
        p = os.path.join(ctx.path(key), rel_)
        shas[f] = sha256_file(p) if os.path.exists(p) else None
    src.update(data=files, data_sha=shas)
    return src


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--paper", help="paper source folder (default: paths.paper_root)")
    ap.add_argument("--out", default=os.path.join(HERE, "claims.json")); ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--report", default="", help="report/reason reference written into the history records")
    args = ap.parse_args()
    cfg = load_config(args.config)
    code_root = cfg["paths"]["code_root"] if os.path.isabs(cfg["paths"]["code_root"]) else os.path.normpath(os.path.join(HERE, cfg["paths"]["code_root"]))
    ctx = Ctx(cfg, "fast", code_root, "/tmp")
    paper = args.paper or ctx.path("paper_root")
    texts = {f: open(os.path.join(paper, f)).read() for f in (M, I, T, R, D, A)}
    ts = prov.tree_sha(code_root, cfg["engine_files"])
    old = {} if args.no_merge or not os.path.exists(args.out) else json.load(open(args.out))
    today = datetime.date.today().isoformat()
    out, errs, pend = {}, [], []
    for sp in SPECS:
        t = texts[sp["file"]]
        try:
            pos, printed = locate(t, sp["near"], sp["printed"], sp.get("nth"), sp.get("sign_slot"))
        except Exception as ex:
            errs.append(f"{sp['id']}: {ex}"); continue
        if sp["id"] in out:
            errs.append(f"{sp['id']}: duplicate id"); continue
        an, off = anchor_for(t, pos, len(printed))
        e = dict(kind=sp["kind"], quantity=sp.get("quantity", ""), status=sp.get("status", "active"), units=sp.get("units", ""),
                 sign_convention="physical", printed=printed, display=sp.get("display", {}),
                 tex=dict(file=sp["file"], anchor=an, offset=off), tests=sp.get("tests", []), history=[])
        if sp.get("recipe"):
            e["recipe"] = dict(name=sp["recipe"], args=sp.get("args", {}))
            rc = sp.get("recompute", {})
            e["recompute"] = dict(fast=dict(tol_rel=1e-9, **rc.get("fast", {})), full=rc.get("full", dict(tol_rel=1e-9)))
        for k in ("producer", "successor", "lit", "check", "notes", "human", "g1"):
            if sp.get(k) not in (None, "", {}):
                e[k] = sp[k]
        ctx.memo.pop("_access", None)
        if "value" in sp:
            e["value"] = sp["value"]
        elif sp.get("recipe"):
            try:
                e["value"] = CR.RECIPES[sp["recipe"]](ctx, **sp.get("args", {}))
            except CR.PendingData as ex:
                e["value"] = None; pend.append(f"{sp['id']}: {ex}")
                if e["status"] == "active":
                    e["status"] = "pending_rerun"
        if e["status"] in ("pending_rerun", "pending_full") and e.get("value") is not None:
            e["status"] = "active"                     # the data exist: the claim is decided
        if sp.get("status") == "pending_rerun" and e.get("value") is None:
            e["status"] = "pending_rerun"
        src = _sources(ctx, sp, e)
        if src:
            e["source"] = src
        if sp["kind"] == "method_param":
            e.setdefault("producer", {})["tree_sha"] = ts
        o = old.get("claims", {}).get(sp["id"]) or old.get("retired", {}).get(sp["id"])
        if o is not None:
            e["history"] = list(o.get("history", []))
            ch = {k: (o.get(k2), e.get(k2)) for k, k2 in (("value", "value"), ("printed", "printed"), ("status", "status"))
                  if o.get(k2) != e.get(k2)}
            if o.get("tex", {}).get("anchor") != e["tex"]["anchor"]:
                ch["anchor"] = ("(re-anchored)", "")
            if (o.get("recipe") or {}).get("name") != (e.get("recipe") or {}).get("name"):
                ch["recipe"] = ((o.get("recipe") or {}).get("name"), (e.get("recipe") or {}).get("name"))
            if ch:
                e["history"].append(dict(date=today, old=o.get("value"), new=e.get("value"), old_printed=o.get("printed"),
                                         new_printed=e["printed"], old_status=o.get("status"), new_status=e["status"],
                                         changed=sorted(ch), reason=REASON, report=args.report))
        else:
            e["history"].append(dict(date=today, event="created", new=e.get("value"), new_printed=e["printed"], new_status=e["status"],
                                     reason=REASON, report=args.report))
        out[sp["id"]] = e
    if errs:
        print("\n".join(errs)); raise SystemExit(1)
    retired = dict(old.get("retired", {}))
    for cid, o in old.get("claims", {}).items():
        if cid in out:
            continue
        o = dict(o); o["history"] = list(o.get("history", []))
        rb = REPLACED_BY.get(cid)
        o["history"].append(dict(date=today, event="retired", old=o.get("value"), old_printed=o.get("printed"), old_status=o.get("status"),
                                 reason=(f"statement replaced by '{rb}' in the final text" if rb and rb in out else "text removed in the final text"),
                                 report=args.report))
        o["status"] = "retired"
        retired[cid] = o
    for cid in list(retired):
        if cid in out:
            retired.pop(cid)
    schema = dict(old.get("_schema", {})) or dict(version=1, macro="\\newcommand{\\claim}[2]{#2}")
    schema.update(
        version=2,
        kinds=["scalar", "range_min", "range_max", "sign", "count", "fit_coef", "method_param", "literature", "assertion", "derived"],
        status=dict(active="value from the current final source", provisional="value from v6 data (pre-fix engine); replace from 'successor'",
                    pending_rerun="value produced by a v7 run still in the queue or only partial in this copy; G2 reports STALE until filled",
                    pending_full="convergence claim, decided by the full-mode C tests", retired="statement no longer in the text (kept in 'retired' with its history)"),
        display="mode round (default; rounding half_even, or half_away when display.rounding says so) | floor/ceil (step) | less_than | greater_than | "
                "at_least_word | order (exponent) | order_of (lo, hi) | sci | word | sign; scale divides the value; decimals; sign auto|always",
        source="data = list of 'pathkey:file' read by the recipe, data_sha = sha256 of each at registration (G1 checks them)",
        g1="'full': the cloud copy of the data is partial, G1 checks the claim only in full mode (on the Mac)",
        history="every change of value, printed text, status, anchor or recipe appends {date, old, new, old_printed, new_printed, old_status, new_status, changed, reason, report}; created/retired events too")
    reg = dict(_schema=schema, built=datetime.datetime.now().isoformat(timespec="seconds"), tree_sha=ts,
               paper=dict(root=os.path.abspath(paper), shas={f: hashlib.sha256(texts[f].encode()).hexdigest() for f in texts}),
               claims=out, retired=retired)
    json.dump(reg, open(args.out, "w"), indent=1, ensure_ascii=False)
    print(f"{os.path.basename(args.out)}: {len(out)} claims; status counts:",
          {s: sum(c['status'] == s for c in out.values()) for s in ("active", "provisional", "pending_rerun", "pending_full")},
          f"; retired {len(retired)}")
    for p_ in pend:
        print("  pending:", p_)


if __name__ == "__main__":
    main()
