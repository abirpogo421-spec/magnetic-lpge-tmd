# How the numbers in the magnetic-LPGE paper are verified

For Jyesta (and any referee). You do not need to read code to follow this. Every check named here has an id (R4, H1, ...) that appears in the dated reports in `reports/`.

## 1. What is guaranteed and what is not

**Guaranteed, if the latest report says PASS:**

- The production engine (`pub_pipeline/lpge.py`, `tb_core.py`, `tmd_params.py`, and the two-band code `kp/kp2.py`) computes the paper's formulas. An independent reference implementation agrees with it on identical k-points to 1e-6 or better. That reference was written from the paper and Liu et al. only, without reading the engine.
- The run scripts that produced the figure data store what they claim to store: the right sign, the right field mirroring, the right b. Short "smoke" copies of each script are rerun and compared with the reference.
- Every number in the paper source is either registered in `claims.json`, with its source, producing script and tolerance, or listed as non-numerical in `numbers_allowlist.json` (for example "four materials" or "B = 0"). A registered number is recomputed or reloaded, rounded, and compared with the digits printed in the paper.
- The planted-bug test passes: all 42 deliberately wrong versions of the code in `mutation/mutants.json` are caught by a physics or claims test (not only by the literal tripwire T1). That covers the 10 bug classes we worried about, plus the gaps found by an independent red team, whose own 36 mutants were rerun against the final suite (section 7).

**Not guaranteed (section 9):** that the Liu three-band model is the right physics, that the literature inputs are true (we only check that they were copied and interpreted correctly), and that fast mode certifies the production digits. Only full mode on the Mac does that.

## 2. How to run it and how to read the report

From `code/verification/` in this repository:

```
python3 verify_all.py --fast                       # about 3 min on the cloud machine (2 cores)
python3 verify_all.py --fast --accept-known "why"  # record today's FAILs as known findings (after review only)
caffeinate -i nohup python3 verify_all.py --full --jobs 10 &     # hours: production grids, convergence
python3 verify_all.py --mutants                    # planted-bug test, about 16 min on 2 cores
python3 verify_all.py --ledger-legacy              # hash the engine files; run now and when the queue ends
```

Each run writes `reports/YYYY-MM-DD_HHMM_<mode>.md` and a `.json` copy. The first table lists only the checks that did not pass. Each row gives the id, what was compared, the measured value, the tolerance fixed in advance, and the status:

| status | meaning |
|---|---|
| PASS | measured value is within the tolerance |
| FAIL | it is not. A FAIL is either a bug or a wrong number in the paper. The detail column says which. |
| KNOWN | a documented finding that is exactly as recorded in `known_failures.json` (today: C8 hotspots in the prod7 maps; resolved baselines move to `resolved`). If anything about it changes, for example one more wrong digit or a different script value, the check is FAIL again and the detail lists only what changed. |
| STALE | the check cannot be decided yet: data from the old (v6) engine, a pending rerun, or an anchor that moved |
| WARN | a known issue that does not change any number, for example a stored array without a declared sign convention |
| SKIP / INFO | not applicable here (no figures in this copy), or a value reported for human reading |

Any FAIL makes the exit code nonzero. KNOWN does not, but it is listed in its own table at the top of the report, so it cannot hide new damage. In `--full` mode, SKIP and STALE also count as failures.

The paths come from `config.json`. Its defaults expect the layout of this repository (`code/pub_pipeline`, `code/kp`, `code/v7jobs`).

## 3. The number chain

```
literature (Liu tables, Kormanyos tables)                 literature/*.json, checked by L1-L5, T2
   -> engine (H(k), geometry, eta)                        checked by T, U, H, R, I, C
   -> run script (edge_v7, maps_v7, ...)                  checked by S1-S6 (smoke copies vs reference)
   -> data file + provenance (engine hash, parameters)    checked by P1-P5 and the engine ledger
   -> claims.json entry (value, source, tolerance)        checked by G1, G4, G5, G6, G7
   -> \claim{id}{printed} in the .tex                     checked by G2, G3 (and X0 once tagged)
```

**Worked example: the threshold χ of TNN MoS2, "+1.9" in Sec. III C.**

- **Literature.** L1 shows that the 19 TNN numbers and λ = 0.073 eV in `tmd_params.py` equal an independent transcription of Liu's tables.
- **Engine.** R1 shows that H(k) agrees with the reference to 1e-13 eV, and its field part to 1e-8. R2 and R3 show that Ω, M, D and the velocity shift agree. R4 and R5 show that η and χ on a small disk around K and K' agree to 1e-6 relative to one valley. H1 shows that η(−B) = −η(B) to 1e-12. H7 shows that the engine, fed a pure two-band Dirac Hamiltonian, gives the opposite sign to κ.
- **Script.** S1 reruns `edge_v7.py MoS2 TNN` on a coarse grid. The stored η matches the reference to 2e-9. The stored `chi_SI_per_T` equals an independent odd-extension and fit to 1e-12. The stored h, R, ν and functional equal the arguments.
- **Data.** `edge7_MoS2_TNN.json` records `lpge_sha256 = ff4a692d…`. The ledger maps that hash to the checked engine tree `0c8e2995…` (P2, P5).
- **Claim.** `chi.thr.TNN.GGA.MoS2` in `claims.json` has recipe `edge7_chi`: it reads `chi_SI_per_T` from `v7_out:edge7_MoS2_TNN.json` (sha recorded), rounds it with scale 1e-4 and one decimal (+1.904 → "+1.9"), and G2 compares the result with `\claim{chi.thr.TNN.GGA.MoS2}{+1.9}` in `results.tex`. G1 reloads the file and fails if its sha or the value changed without a history record.
- **Convergence.** In full mode, C2, C3, C6 and C7 measure how much the value moves under grid spacing, disk radius, field fit and ω averaging. Each claim must keep its printed digits under ±2× that movement.

## 4. The test families (all tolerances were fixed before the first run)

| family | what it checks | typical tolerance |
|---|---|---|
| T static | every numerical literal in the code reachable from η is registered (`knobs.json`); literature numbers appear only in their designated tables; the reference imports nothing from the engine; scripts import only the hashed files | exact |
| U units | e³a/ħ², m_e a²/ħ², e a²/ħ from CODATA; μ_B in every file; the valley gap splitting equals 4 μ_B B | 1e-12, 1e-9 |
| L literature | Liu transcription (24 parameter sets); energies at Γ, K, M; spin splitting 2λ at K; Kormányos spin label, A-pair masses and the script's input table; 2m_D − λ = E_A | exact, 1e-12 |
| H symmetry and analytic | oddness in B; zero at B = 0; C3 and mirror relations of the raw tensor; Hermiticity; spin-block structure under time reversal; L_z = ±2 valley character; engine on a pure warped Dirac model (kernel, ring average, sign rule); exact T-pairing of the disks; Lorentzian sum vs brute force (no binning, no window); Hellmann–Feynman; Berry sum rule | 1e-9 to 1e-14 |
| R reference | engine vs the independent reference: H, E, Ω, M, D, g, velocity shift, η for all field-term variants, χ, the C3-symmetrized tensor, Löwdin v and κ, the two-band code | 1e-6 to 1e-13 |
| I invariance | η at one ω does not depend on the other requested ω; chunk size, k order, additivity; ∫η dω independent of ν; no state leaks between calls | 1e-12 |
| S smoke | the production scripts on reduced grids vs the reference. S1 covers TNN GGA, NN GGA and TNN LDA, with each one's own E_g. S4 covers three decomp paths: no map; a finished map with a dominant negative lobe; an unfinished map, which must be refused. S4 also checks the fallback van Hove table against the registered claims. S6 runs `fig_v7.py` and `fig_v7_all.py` on the smoke outputs and reads back what they actually plot: odd mirroring and equality with the stored data for Figs. 7, 8, 9 and 11. | 1e-6 |
| U4, I5 (added after the red team) | the absolute k-space measure (disk dA = h², BZ grid sum = 8π²/√3, disk point sets equal the reference's); a production-shaped call (1001 ω on 51k k-points) gives the same η at one ω as a single-ω call, for any chunk size | exact, 1e-12 |
| C convergence | h rate study; finite-difference step; stored C3 violation; full mode: N, h, R, ν, field fit, ω averaging | 0.3–1% |
| G claims | data reload, printed digits, orphan numbers, recompute, method parameters vs scripts, assertions | exact / 1e-9 |
| P provenance | sidecars or ledger-known engine hashes; figure newer than its data; PDF metadata; P6: the figure PDFs in the paper carry the `fig_v7_all.py` sidecar with the current v7 input shas, and the Fig. 7 legend energies (read with pdftotext) equal the registered van Hove energies | exact |

Five tolerances were changed after the first measurement. In each case the protocol value was below the engine's own finite-difference truncation. The measured value and the reason are in each report row.

| check | protocol tol | used | measured | why |
|---|---|---|---|---|
| R4, no velocity shift | 1e-7 | 1e-6 | 2.3e-7 | central-difference error at step 10⁻⁴ (C5 confirms it scales as EPS²) |
| R4/S2, velocity shift on the full BZ | 1e-6 | 1e-4 | 3.5e-5 | at near-degeneracies of the conduction and remote bands, the derivative of M is limited by both codes' differences. The reference's own step dependence is 7.6e-6. |
| H3, C3 on the raw tensor | 1e-10 | 1e-5 | 9e-7 | x/y central differences are not C3-covariant. The error scales as EPS² down to the round-off floor. |
| R8, kp2 | 1e-5 | 5e-5 | 1.9e-5 | kp2 uses its own step of 10⁻³/Å |
| C0, rate study | monotone, order ≥ 1.5 | or all steps < 1e-5 | steps of 6e-6 and 2e-6 | the h error is already at the noise floor at h = 0.004/a |

## 5. Conventions

- η_phys = −2 · si_factor · η_code, with si_factor = e³a/ħ². The current is that of electrons, charge −e.
- k is in units of 1/a, with x along a metal–metal bond and K = (4π/3a, 0).
- At K the valence state has L_z = +2 and lies in the spin-up block of the engine.
- b = μ_B B, with no ħ. g_s = 2. μ_B = 5.7883818012e-5 eV/T (CODATA 2014) in the engine; the reference uses the CODATA 2018 value (U2 reports the 8.3e-10 difference).
- **Kormányos labels.** Kormányos et al. call the upper valence band at K "↓" ("in the VB the upper index (1) ((2)) is equivalent to ↓ (↑)", p. 20). The engine calls the same band-edge (A) pair spin-up, because the valley and spin conventions differ. L4 checks the physical pair through the masses, not through the arrow.

## 6. History that the suite now guards against

| id | what went wrong | which check catches a recurrence |
|---|---|---|
| V-01 | κ and masses were taken from Kormányos's B-pair (↑) column instead of the band-edge pair | L4 (spin label, masses, script table), T2 (stale tables), G2. The paper's Table II column and the "κ = −1.1 to −1.5" and "+1.1 to +1.9 × 10⁻³" sentences still show the old values: G2 FAIL. |
| V-03 | the v6 engine binned transitions into 0.2 meV bins | H9, R4, T1 (mutant M12) |
| V-04 | the v6 engine dropped transitions outside min(ω) − 40ν … max(ω) + 40ν | H9, I1, R4 (mutant M13) |

**Red-team round (2026-10-14).** An independent red team wrote 36 mutants without seeing ours. The first suite missed 8 real ones. Each is now caught by a physics or claims test, and the same bug class was added to `mutants.json` as M35–M44:

| red-team id | gap | fix |
|---|---|---|
| RT22 | η wrong only for production-shaped calls (more than 200 ω on more than 20k points); every fast test passed | I5 |
| RT18a, RT18b | area element ×√3/2: the reference used the engine's dA, so the error cancelled | the reference builds its own dA; U4 checks the measure |
| RT12a | LDA runs used the GGA Hamiltonian | S1 for TNN LDA, against the LDA reference |
| RT28 | NN runs used the TNN gap | S1 for NN, with each model's own E_g |
| RT29, RT24 | van Hove fallback table typo; signed instead of absolute argmax | S4 (a) table vs claims; S4 (b) negative-lobe fixture |
| (red-team finding) | decomp took vH from an unfinished map, i.e. ω[0] | done-guard in decomp_v7.py; S4 (c) |
| RT11c | figure mirroring shifted by one field step | S6 read-back of the plotted arrays |
| RT26 | queue typo hidden because G6 was already red | known-failure baselines: a changed item makes the check FAIL again |

## 7. Planted-bug test (kill matrix, 2026-10-14, cloud)

The control run on the unmutated code (rerun 2026-10-14 against the final-text registry and the v7 data): 213 checks in the mutation subset (T, U, L, H, R, I, S, C0, C5, G, P), 0 failing, the 4 KNOWN findings unchanged. The 6 STALE checks cannot be decided in a control run and are excluded. A mutant counts as caught only if a non-static test fails; a T1/T2-only kill is reported as 'static only'. Result: 44 mutants, 42/42 non-equivalent ones killed by a physics or claims test; M05 (9.5e-14) and M28 (1.6e-14) are EQUIVALENT-OK. Full table: `mutation/report.md`.

| mutant | planted bug | caught by |
|---|---|---|
| M01 | prefactor sign | R4, H7, H9 |
| M02, M03 | script prefactor sign (maps, decomp) | S2; S4 |
| M04 | swapped valley (L_z sign) | H4–H6, L3, R1, R4, U3, T1 |
| M05 | K and K′ relabelled (**equivalent**) | outputs unchanged; flagged only by the static T1 |
| M06 | broken T-pairing of the disks | H1, H2, H8 |
| M07, M08 | dropped factor a (η and phase space) | U1, R4; U1, R2, T1 |
| M09, M11 | spin-block sign of SOC / orbital Zeeman | H4, H5, L3, R1, R4, U3 |
| M10 | spin-Zeeman sign (equivalent for η) | H4, H5, R1, R2 only: no η-level test, as required |
| M12 | 0.2 meV binning reintroduced | H9, R4, T1 |
| M13 | ±40ν window cut reintroduced | H9, I1, R4, T1 |
| M14 | window cut in kp2 | R8, T1 |
| M15, M16 | ħ wrongly put into b (engine, k·p script) | H4, H5, R1, U3, T1; S6 |
| M17 | missing τ in kp2 | H2, R8 |
| M18 | off-by-one in field mirroring | S1 |
| M19, M20, M34 | wrong λ (table, convention, override) | L1, L3, R1, H4; L3, R1, T1; R5, S5, T1 |
| M21–M25 | velocity-shift, moment-energy, phase-space band, Berry sign, factor 2 in M | R3, R4; R4; R4; R2, R4, T1; R2, R4, T1 |
| M26 | finite-difference step 10⁻² | H10, R2, T1 |
| M27 | Lorentzian normalization | H9, R4, T1 |
| M28 | (b,c) symmetrization dropped (**equivalent**) | outputs unchanged at 4e-14; flagged only by T1 |
| M29 | C3 symmetrizer angle | R6, T1 |
| M30 | hopping typo (+0.001 eV) | L1, L2, H4 |
| M31, M32, M33 | SOC counted twice in the k·p gap; wrong Kormányos column; field normalization | L5; L4, T2; S5 |
| M35 | sub-blocks sliced with CHUNK instead of lchunk (RT22) | I5 |
| M36, M37 | area element ×√3/2 (disks; full BZ) | R4, U4 (+T1) |
| M38, M39 | LDA run with the GGA H; NN run at the TNN gap | S1 |
| M40, M41, M44 | fallback vH typo; signed argmax; unfinished map accepted | S4 |
| M42 | figure mirroring off by one | S6 |
| M43 | queue grid typo (0.007) | G6 (`param.edge.h.methods` vs `run_v7_queue.sh`; before the final text G6 was red and the known-failure baseline caught it) |
| X1–X4 | printed digit edited; untagged decimal added; data rerun without replot; registry value edited without history | G2; G3; P2; G1 |

The red team's own 36 mutants, rerun on the final suite (`mutation/redteam_kill_table.md`): 34 of 34 non-equivalent ones are killed by a physics or claims test, none by T1 alone. The two that survive are equivalent by construction: RT11a (a duplicated B = 0 row adds nothing to the least-squares fit) and RT25 (transposed C3 rotations; the rotation set is closed under inversion). Eight kills still rest on a single check: RT05, RT09a, RT20 on S6; RT11b, RT12a, RT28 on S1; RT12c, RT27 on S5. Each of those is the only place where that script runs.

After the final-text update (2026-10-14) the red-team mutants whose killing checks changed were rerun: RT24 (S4 negative-lobe fixture), RT26 (G6, now without a baseline) and RT29 (S4 fallback table, now compared within 5 meV because the table holds the v6 energies) are all killed. `rt.py` still builds its paper copy from the pre-stage-2 pull, so its control now has 3 FAILs (claims checks against the old text); point it at the tagged stage-2 source for a clean control.

## 8. Rules for changing code, data or the paper

1. Do not edit any production script while the v7 queue is running.
2. After any change in `pub_pipeline/`, `kp/`, `v7jobs/`, `literature/` or `claims.json`, run `verify_all.py --fast`. Use outputs only from a tree with a passing `verified.lock`.
3. A change to a test means rerunning `--mutants`.
4. A literal edit makes T1 fail until `--register-knobs` is run and the diff of `knobs.json` is reviewed. T1 is a tripwire, not evidence: the mutation harness counts a mutant as caught only when a physics or claims test fails too.
5. A finding may be accepted as KNOWN only with a reason (`--accept-known "reason"`), and only in `known_failures.json`. Accept again whenever the finding changes; fix it, and the WARN tells you to drop the baseline.
6. Changing a registered value requires a `history` entry (old, new, reason, report). Without one, G1 fails.
7. New numbers in the paper must be registered as claims or allow-listed. Otherwise G3 fails.
8. When the queue finishes:
   - run `tools/fill_pending.py --write`, which turns pending claims into values, each with a history entry;
   - run `--fast`;
   - fix the paper where G2 says it must change;
   - tag the paper with `tests/tag_tex.py` and check that the PDF text is unchanged with `tools/x0_check.py`.

## 9. Literature sign-off

`literature/liu_tables.json` is the reference's independent transcription of Liu Tables II–IV. `literature/kormanyos_2015.json` holds verbatim rows of Kormányos Tables 3, 4 and 6 (DFT columns), with page numbers and the p. 20 spin-label sentence. A human should compare both files against the printed papers once. The DFT/GW choice and the A-pair conduction-band row per material are interpretation choices, recorded under `interpretation`.

## 10. Honest limits

- **Model validity is not tested.** That covers Liu's three-band optics, the absence of excitons, constant ν and T = 0. What is done instead: the NN/TNN/GGA/LDA/k·p spread is reported, and C4 (full mode) shows which signs survive a 4× change of ν.
- **Literature is checked for transcription and interpretation, not for truth.** A small hopping error passes L2; only the double entry of L1 catches it. Liu's a²/2 convention (V-34) is still to be checked against the print.
- **Shared formulas.** Both codes implement the paper's equations, and both were written by AI. H7, U3, H5 and H10 test against physics that does not depend on those formulas. A human should still read the roughly 30 key formula lines of `ref/refimpl.py`.
- **Mutation testing proves sensitivity to the planted classes only.** M05, M10 and M28 are invisible at the η level by construction. Static T1 flags any literal edit, so it "catches" equivalent edits too. That is intended, but it does not prove anything about physics.
- **Fast mode cannot certify production digits.** Only the full-mode checks C1–C7 and G5 do, on the Mac.
- **Figs. 1–6 have no sidecars** (v6 figure scripts): P1/P4 STALE for them.
- **Expensive claim recipes** (`psf_thr`: the engine at threshold, about 3 min; `kp_lam_change`: kp2 at Nk = 401, about 1 min each) are not recomputed in fast mode (G4 INFO); G5 recomputes them in full mode.
- **Figure read-back covers Figs. 7, 8, 9 and 11.** Figs. 10 and 12 are drawn by the same script but are not mirrored. Their smoke inputs (8 chiw and 4 λ runs) were left out to keep fast mode under 4 min. The full set runs on real data once the queue is done.
- **The decomp done-guard was replicated in the cloud copy of `decomp_v7.py`**, from the description of the Mac version. Its hash (`db2a0ba3…`) must be re-recorded on the Mac with `--ledger-legacy`, which prints CHANGED scripts and keeps a `script_history`.
- **The M05 equivalence margin is thin**: its fixture difference is 9.5e-14 against a 1e-13 limit, and it is pure summation-order round-off. It may cross the limit on another BLAS.
- **Wording, axis labels, and numbers typed into Overleaf after a run** are caught on the next run (G3), not prevented.
- **The allow-list was proposed by the scanner** (102 entries for the final text: 89 kept, 13 new, 30 stale ones dropped; all `status: proposed`). Abir needs to read and approve each entry (G3 WARN).

## 11. The registry for the final text (2026-10-14)

`claims.json` (schema 2) registers the final text: the stage-2 source (live Overleaf + 82 audit edits + 29 v7 edits), tagged in a copy with `tests/tag_tex.py`; `pdftotext` of the tagged and untagged builds is identical (X0, 19 pages). It holds 251 claims, all active (the TNN LDA runs are complete since the final text, 2026-10-14 08:46). Eight claims of the older text were retired, each with a history record that names its successor (for example `fig7.linear20` → `fig7.lin5`). Every claim kept from the earlier registry has a history record of the change (old and new value, printed text, status, anchor, recipe).

Every value is computed from the v7 outputs, not from `numbers_v7.json` (which `fig_v7_all.py --only` overwrites), using the definitions of the text:

| quantity | definition (recipe) | source |
|---|---|---|
| van Hove energy | ω of max \|η\| at 60 T (`vh7_energy`) | `prod7_{m}_F.npz` (finished maps only) |
| van Hove χ | η(1 T)/1 T (`vh7_slope`) | `prod7` |
| "within 5% up to 20 T", "3–29% below at 60 T" | relative to that initial slope (`vh7_dev20`, `vh7_below60`) | `prod7` |
| threshold χ | `chi_SI_per_T` of the edge run (fit sB + cB³) (`edge7_chi`) | `edge7_{m}_{model}.json` |
| Fig. 11 fit, fit at the first-principles κ | linear fit of the 8 GGA TB χ against the Löwdin κ of the reference; κ_↓ of Kormányos Table 6 (`fit11`) | `edge7`, `ref/`, `literature/` |
| N and h convergence | max \|rel\| (`conv_c2`, `conv_c3`) | `v7_checks:c2_N_convergence.json`, `c3_h_convergence.json` |
| D vs Z, F/Z | `decomp7_ratio_range` | `decomp7_{m}.json` |
| Fig. 10 sign changes and persistence | 5 meV running mean, first zero crossing, window end 0.45 eV (`chiw7_*`) | `chiwR9v7_*` |
| Fig. 12 R² | linear fit over [0.5, 1.5] λ₀ at E_g + 0.1 eV, code sign × (−2) (`lam7_R2`) | `lam7_{m}_F.npz` |
| TB band warping (−0.7; −0.8 to −1.0) | Kormányos-style 16% fit and exact q³cos3φ coefficient of the reference bands (`tb_warp`) | `ref/` |
| phase-space factor at threshold (5–22%) | engine, E vs ED at E_g + 0.01 eV, disks 0.9/a, h = 0.002/a (`psf_thr`) | engine |
| k·p λ sweep (3.2%, 8%) | kp2 recompute, Nk = 401 (`kp_lam_change`) | `kp/kp_params_TNN.json` |

The Mac paths are the defaults in `config.json` (`v7_out = ../v7jobs`, `v7_checks = ../v7jobs/checks`, `v7_figs = ../v7jobs/figs_v7`, `v6_code = ../../paper/v6_work/code` and `v6_jobs = ../v6jobs` for the two v6 figure scripts that G6 reads, and `paper_root`, which must be set to the tagged Overleaf copy). The `c2`/`c3`/`c4` check files must be copied to `v7jobs/checks/` on the Mac (their producer is not in the queue). In the cloud, `config_cloud.json` points `v7_out` at a read-only symlink view (`/home/claude/verif/v7view`) of `/home/claude/v7data` and `/home/claude/v7data2`, and `paper_root` at `/home/claude/v7p/stage2_tagged`. To rebuild after a text edit: `python3 tools/build_claims.py --config config.json --paper <untagged source>` (keeps and extends the history), then `tests/tag_tex.py <copy>`, `tools/x0_check.py`, and `verify_all.py --fast`.

Results on the tagged final copy (cloud, `reports/2026-10-14_0907_fast.md`): PASS. G1–G7 pass (G3 WARN: allow-list still proposed), P6 passes with the v7 figure PDFs and the Mac `decomp7` files. The two findings of the first pass (phase-space 5–22% → 5–40%, NN 1.5% → about 1.5% vs the zero-field slope) and the v6 figure PDFs were fixed in the text; their baselines are in `known_failures.json` → `resolved`. Only C8 remains KNOWN.
