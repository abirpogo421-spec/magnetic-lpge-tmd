# Mutation report 2026-10-14T08:24:51

- verdict: **PASS**; killed by a physics/claims test: 42/42 non-equivalent mutants; control: 213 checks, 0 failing (undecidable in control and excluded: 6 STALE); wall 185 s
- subset: `T,U,L,H,R,I,S,C0,C5,G,P`; a kill by the static tripwires T1-T4 alone is reported as 'KILLED (static only)' and does not count

| mutant | file | class | expected killers | status | actual killers | note |
|---|---|---|---|---|---|---|
| M01 | pub_pipeline/lpge.py | prefactor sign | R4, H7 | KILLED | H7, H9, R4 | phase A |
| M02 | v7jobs/maps_v7.py | prefactor sign (script) | S2 | KILLED | S2 | phase A |
| M03 | v7jobs/decomp_v7.py | prefactor sign (script) | S4 | KILLED | S4 | phase A |
| M04 | pub_pipeline/tb_core.py | swapped valley | H6, U3, R1, R4 | KILLED | H4, H5, H6, L3, R1, R4, T1, U3 | phase A |
| M05 | pub_pipeline/lpge.py | valley label swap |  | EQUIVALENT-OK | T1 | fixture outputs rel diff 9.5e-14 (<= 1e-13 required); eta-level failures: none; flagged by static checks ['T1'] (a literal edit always needs re-registration) |
| M06 | pub_pipeline/lpge.py | broken T-pairing | H8, H2, H1 | KILLED | H1, H2, H8 | phase A |
| M07 | pub_pipeline/lpge.py | dropped factor a | U1, R4 | KILLED | R4, U1 | phase A |
| M08 | pub_pipeline/tb_core.py | dropped factor a (phase space) | U1, R2 | KILLED | R2, T1, U1 | phase A |
| M09 | pub_pipeline/lpge.py | spin-block sign (SOC) | H5, R1, R4 | KILLED | H4, H5, L3, R1, R4, U3 | phase A |
| M10 | pub_pipeline/lpge.py | spin-block sign (spin Zeeman); equivalent for eta | H5, R1 | KILLED | H4, H5, R1, R2 | Hamiltonian-level killers ['H4', 'H5', 'R1']; eta-level killers none -> OK (equivalent for eta, needs the H-level tests) |
| M11 | pub_pipeline/lpge.py | spin-block sign (orbital Zeeman) | H5, R1, R4 | KILLED | H4, H5, R1, R4, U3 | phase A |
| M12 | pub_pipeline/lpge.py | binning reintroduced | H9, R4, T1 | KILLED | H9, R4, T1 | phase A |
| M13 | pub_pipeline/lpge.py | window cut reintroduced | H9, I1, I3, R4 | KILLED | H9, I1, R4, T1 | phase A |
| M14 | kp/kp2.py | window cut (k.p) | T1, I1, R8 | KILLED | R8, T1 | phase A |
| M15 | pub_pipeline/lpge.py | wrong hbar in b | U3, R1 | KILLED | H4, H5, R1, T1, U3 | phase A |
| M16 | kp/kp_fp_all_v7.py | wrong hbar in b (k.p) | S6 | KILLED | S6 | phase A |
| M17 | kp/kp2.py | missing tau | R8, H2 | KILLED | H2, R8 | phase A |
| M18 | v7jobs/edge_v7.py | off-by-one field mirroring | S1 | KILLED | S1 | phase A |
| M19 | pub_pipeline/tmd_params.py | wrong lambda | L1, L3, R1 | KILLED | H4, L1, L3, R1, T2 | phase A |
| M20 | pub_pipeline/tb_core.py | lambda convention (2 lambda) | L3, R1 | KILLED | H4, L3, R1, T1 | phase A |
| M21 | pub_pipeline/lpge.py | velocity-shift sign | R3, R4 | KILLED | R3, R4 | phase A |
| M22 | pub_pipeline/lpge.py | moment-energy sign | R4 | KILLED | R4 | phase A |
| M23 | pub_pipeline/lpge.py | phase-space weight on wrong band | R4 | KILLED | R4 | phase A |
| M24 | pub_pipeline/lpge.py | Berry curvature sign | R2, R4 | KILLED | R2, R4, T1 | phase A |
| M25 | pub_pipeline/lpge.py | factor 2 in orbital moment | R2, R4 | KILLED | R2, R4, T1 | phase A |
| M26 | pub_pipeline/lpge.py | finite-difference step | R2, C5, T1 | KILLED | H10, R2, T1 | phase A |
| M27 | pub_pipeline/lpge.py | Lorentzian normalization | H9, R4 | KILLED | H9, R4, T1 | phase A |
| M28 | pub_pipeline/lpge.py | drop (b,c) symmetrization; g already symmetric |  | EQUIVALENT-OK | T1 | fixture outputs rel diff 1.6e-14 (<= 1e-13 required); eta-level failures: none; flagged by static checks ['T1'] (a literal edit always needs re-registration) |
| M29 | pub_pipeline/tb_core.py | C3 symmetrizer angle | R6, H3 | KILLED | R6, T1 | phase A |
| M30 | pub_pipeline/tmd_params.py | hopping typo (TNN t2 +0.001 eV) | L1 | KILLED | H4, L1, L2 | phase A |
| M31 | kp/kp_fp_all_v7.py | SOC counted twice in gap | L5 | KILLED | L5 | phase A |
| M32 | kp/kp_fp_all_v7.py | wrong Kormanyos spin column (V-01) | L4, T2 | KILLED | L4, T2 | phase A |
| M33 | v7jobs/lam_v7.py | field normalization | S5 | KILLED | S5 | phase A |
| M34 | pub_pipeline/lpge.py | wrong lambda (override path) | R5, S5 | KILLED | R5, S5, T1 | phase A |
| M35 | pub_pipeline/lpge.py | large-grid-only accumulation (red team RT22) | I5 | KILLED | I5 | phase A |
| M36 | pub_pipeline/lpge.py | area element sqrt3/2, disks (RT18a) | U4, R4 | KILLED | R4, T1, U4 | phase A |
| M37 | pub_pipeline/tb_core.py | area element sqrt3/2, full BZ (RT18b) | U4, R4 | KILLED | R4, T1, U4 | phase A |
| M38 | v7jobs/edge_v7.py | LDA runs use the GGA H (RT12a) | S1 | KILLED | S1 | phase A |
| M39 | v7jobs/edge_v7.py | E_g from TNN for NN runs (RT28) | S1 | KILLED | S1 | phase A |
| M40 | v7jobs/decomp_v7.py | fallback vH table typo (RT29) | S4 | KILLED | S4 | phase A |
| M41 | v7jobs/decomp_v7.py | vH = argmax of signed eta (RT24) | S4 | KILLED | S4 | phase A |
| M42 | v7jobs/fig_v7.py | figure mirroring off by one (RT11c) | S6 | KILLED | S6 | phase A |
| M43 | v7jobs/run_v7_queue.sh | queue grid typo hidden by a red check (RT26) | G6 | KILLED | G6 | phase A |
| M44 | v7jobs/decomp_v7.py | partial prod7 accepted (red team sec. 3) | S4 | KILLED | S4 | phase A |

| non-code | status | killers | change |
|---|---|---|---|
| X1 | KILLED | G2 | Table II v(MoS2) 4.93 -> 4.94 in a copy of theory.tex (the protocol's +1.9 -> +1.8 is a pending claim until the rerun) |
| X2 | KILLED | G3 | untagged '0.123' inserted in results.tex |
| X3 | KILLED | P2 | prod7 data newer than lpge_map_grid.pdf |
| X4 | KILLED | G1 | tab5.lpge.B30 value x1.01 in a copy of claims.json |
