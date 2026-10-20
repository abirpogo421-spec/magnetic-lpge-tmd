# Red-team mutants rerun against the final suite (2026-10-14, cloud)

Source: /home/claude/redteam (mutants_rt.py, rt.py, summarize.py, unchanged), driven by tools/run_redteam.py. Control: PASS (191 PASS, 3 KNOWN, 0 FAIL).

| id | class | result | killed by | masked |
|---|---|---|---|---|
| RT01 | eta prefactor sign | KILLED | H7, H9, R4, R5, R6, S1, S2, S3, S4, S5 |  |
| RT02 | swapped valley (disk position) | KILLED | C0, H1, H2, H8, S1, T1, U4 |  |
| RT03 | swapped valley (kp2 spin-valley pairing) | KILLED | H2, S6, T1 |  |
| RT04 | dropped/wrong a (SI) | KILLED | R4, R5, S1, U1 |  |
| RT05 | dropped a (k.p SI) | KILLED | S6 |  |
| RT06 | spin-block SOC sign | KILLED | H4, L3, R1, R2, R3 |  |
| RT07 | binning reintroduced | KILLED | S2, S4 |  |
| RT08 | energy-window cut reintroduced | KILLED | H9, R4, R5, R6, S1, S3, S4, S5 |  |
| RT09a | hbar factor in b | KILLED | S6 |  |
| RT09b | hbar factor in b | KILLED | R2, R4, R5, R6, S1, S2, S3, S4, S5, T1 |  |
| RT10 | missing valley factor tau | KILLED | G4, H2, R8, S6 |  |
| RT11a | field mirroring off-by-one | SURVIVED |  |  |
| RT11b | field mirroring off-by-one | KILLED | S1 |  |
| RT11c | field mirroring off-by-one | KILLED | S6 |  |
| RT12a | wrong lambda / parameter set | KILLED | S1 |  |
| RT12b | wrong lambda | KILLED | H4, L1, L3, R1, R2, R3, R4, R5, R6 |  |
| RT12c | wrong lambda (override) | KILLED | S5 |  |
| RT13 | subtle: D on wrong band | KILLED | R2, R4, R5, R6, S1, S2, S3, S4, S5 |  |
| RT14 | subtle: velocity-shift sign | KILLED | R3, R4, R5, R6, S1, S2, S3, S4, S5 |  |
| RT15 | subtle: nu as FWHM | KILLED | H9, R4, R5, R6, S1, S2, S3, S4, S5, T1 |  |
| RT16 | subtle: m_e in kappa_orb | KILLED | R2, R3, R4, R5, S1, S3, S4, S5, T1, U1 |  |
| RT17 | subtle: C3 symmetrizer | KILLED | R6, S2, S4, T1 |  |
| RT18a | subtle: area element sqrt3/2 (disks) | KILLED | R4, R5, R6, S1, S3, S4, S5, T1, U4 |  |
| RT18b | subtle: area element sqrt3/2 (full BZ) | KILLED | R4, S2, S4, T1, U4 |  |
| RT19 | subtle: hopping typo | KILLED | L1, L2 |  |
| RT20 | subtle: k.p mv sign | KILLED | S6 |  |
| RT21a | subtle: stale number in .tex | KILLED | G2 |  |
| RT21b | subtle: stale number in .tex | KILLED | G2, G3 |  |
| RT22 | extra: large-grid-only accumulation bug | KILLED | I5 |  |
| RT23 | extra: float32 accumulator | KILLED | C0, H1, H2, H9, I2, I5, R4, S1, S4 |  |
| RT24 | extra: vH energy pick | KILLED | S4 |  |
| RT25 | extra: C3 rotation sense (expected equivalent) | SURVIVED |  |  |
| RT26 | extra: run-queue grid typo | KILLED | G6 |  |
| RT27 | extra: lambda-sweep -B leg | KILLED | S5 |  |
| RT28 | extra: script gap model hard-coded | KILLED | S1 |  |
| RT29 | extra: fallback-table typo | KILLED | S4 |  |

34/34 non-equivalent mutants killed by a physics/claims test (none by T1 alone). RT11a (duplicated B=0 row, zero contribution to lstsq) and RT25 (transposed C3 rotations; the group is closed under inversion) are equivalent and survive by construction.
