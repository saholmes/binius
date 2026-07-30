# Modifications to Binius (Apache-2.0 derivative work)

This repository is a fork of **Binius** (upstream:
`https://gitlab.com/IrreducibleOSS/binius`), Copyright Irreducible Inc., licensed
under the **Apache License, Version 2.0** (see `LICENSE.txt`). This file records
the changes made in this derivative work, as required by Apache-2.0 Section 4(b)
("You must cause any modified files to carry prominent notices stating that You
changed the files"). Each modified source file also carries an in-file
`Modified ... by Anonymous` header.

## Purpose

Generalize the challenge / extension field (`FExt`) above the stock
tower-level-7 (`2^128`) ceiling, so real STARK proofs verify at NIST security
levels **L1 / L3** (a `2^256` challenge field, tower level 8) and **L5**
(`2^512`, tower level 9). The DP24 ring-switch soundness argument is general over
the extension degree, so these are **implementation generalizations, not changes
to the security argument**. The stock `2^128` path is byte-for-byte unchanged and
Binius's own test suite passes (`binius_core` 138/138, `binius_m3` unit tests).

## Modified files

- `crates/core/src/ring_switch/tower_tensor_algebra.rs`
- `crates/core/src/ring_switch/prove.rs`
- `crates/core/src/ring_switch/verify.rs`
  Replaced the hardcoded tower-level-7 kappa dispatch literals `{0,1,2,3,4,7}`
  with type-derived match guards (`k == TensorAlgebra::<Tower::Bx, FExt>::kappa()`),
  so the ring-switch reduction admits an `FExt` at any tower level (level 7 is
  bit-identical; level 8 / 9 auto-yield `{0,2,3,4,5,8}` / `{0,3,4,5,6,9}`).

- `crates/core/src/constraint_system/prove.rs`
  Made the base-field tower-level dispatch `FExt`-relative
  (`FExt::TOWER_LEVEL` instead of the hardcoded `7`) and added a level-8 arm.

- `crates/m3/src/gadgets/hash/keccak/mod.rs`
  Made the Keccak-f gadget generic over its top field (relaxed the concrete
  `BinaryField128b` scalar bounds; dropped a vestigial, never-invoked
  `PackedTransformationFactory` bound).

- `crates/m3/src/builder/table.rs`
  `add_constant` builds its transparent constant polynomial over the top field's
  own underlier rather than a hardcoded 128-bit underlier.

- `crates/core/src/merkle_tree/hiding.rs` (new file)
  A1 of the zero-knowledge roadmap: an opt-in **hiding (salted) binary Merkle
  commitment** layered over the unmodified `BinaryMerkleTreeProver`. Appends a
  per-leaf salt as extra committed field elements so the leaf digest
  `H(real ‖ salt)` is hiding, with no change to the Merkle scheme; the salt is
  revealed as part of the opened values. The default non-hiding path is untouched.

- `crates/core/src/merkle_tree/mod.rs`
  Register `mod hiding;` and re-export `salt_leaves` / `leaf_values`.

- `crates/m3/src/gadgets/masking.rs` (new file)
  A4 of the zero-knowledge roadmap: `MaskColumns`, an opt-in gadget that adds N
  committed, unconstrained masking columns to a table and fills them with fresh
  randomness (via a caller-supplied fill closure, so the gadget carries no RNG
  dependency). The randomness layer for a masked commitment / zero-knowledge
  sumcheck. Existing gadgets/tables are unaffected.

- `crates/m3/src/gadgets/mod.rs`
  Register `pub mod masking;`.

- `crates/core/src/protocols/fri/zk.rs` (new file)
  A2 of the zero-knowledge roadmap (port of Diamond, IACR ePrint 2025/1015,
  Construction 4.1). Two opt-in coefficient-domain helpers:
  - `pad_message_high` — technique-1 "high-end padding": lay out the committed
    coefficient vector as `[message ‖ kappa random ‖ zeros]` over the doubled
    `ell -> ell+1` novel-basis domain, so `Z_H(X)·r(X)` randomises the FRI query
    domain while leaving the multilinear evaluation on `H` unchanged.
  - `sample_mask_poly` + `combine_masked` — technique-2 second-random-polynomial
    combination: sample a fully-random `f'` of the padded length and form the virtual
    `alpha·f + f'` (α a large-field verifier challenge) that the FRI low-degree test
    runs on. F-linearity of RS encoding makes coefficient-domain combination identical
    to codeword combination; proximity on `alpha·f + f'` certifies both `f` and `f'`
    are low-degree while the random `f'` hides `f` at the opened points.
  - `extend_multilinear_zk` — the evaluation-domain (BaseFold) view of technique-1
    padding: takes the `2^ell` hypercube evaluations of the committed multilinear `g`
    and returns the `2^{ell+1}` evaluations of the extension `G` (new variable most
    significant) with `G(x,0)=g(x)` and `kappa` random masks in the `G(x,1)` slice. The
    accompanying test drives binius_math's own `MultilinearExtension` evaluator to
    confirm the load-bearing invariant — `G(z,0)=g(z)` for every proven point while the
    `=1` slice is exactly the live mask — so the padding is validated through the real
    evaluation layer, not a stand-in.
  A further test (`fri_fold_is_linear_...`) validates technique-2's enabling invariant
  against the real `fold_codeword`: `fold(alpha*f + f') = alpha*fold(f) + fold(f')`, so
  a verifier can open the separately-committed `f` and `f'` trees and reconstruct the
  combined fold at the query positions — the soundness basis for the two-tree opening.
  Opt-in; the default (non-hiding) FRI commitment path is untouched. The remaining A2
  step is threading these helpers through the interleaved sumcheck-FRI prover/verifier
  in `crates/core/src/piop/{prove,verify}.rs` (message dimensioning + the final
  consistency check) for a full zero-knowledge `commit`/`prove`/`verify`.

- `crates/core/src/protocols/fri/mod.rs`
  Register `mod zk;` and re-export `pad_message_high`, `sample_mask_poly`,
  `combine_masked`, `extend_multilinear_zk`.

- `crates/core/src/piop/zk.rs` (new file)
  A2 protocol-level wiring into the FRI-Binius PCS compiler (`crate::piop`):
  `mask_multilinear`, `mask_n_vars`, `augment_commit_meta_with_mask`. The prover
  appends a freshly-sampled masking multilinear to the committed batch, in its own
  largest-variable bucket. Because that bucket carries no sumcheck claim it is an
  unconstrained committed column — a shape the existing `prove`/`verify` already
  support — so this needs NO change to `prove`, `verify`, `FRIParams`, or the verifier
  consistency check. The mask enters the merged FRI message, so the commitment and
  codeword are randomised (commitment hiding). The tests run the full
  commit/prove/verify end-to-end and check that independent masks yield unlinkable
  commitments while every real opening still verifies. Scope: this is *commitment*
  hiding (the A4 masking-column technique at PCS scope); opened-value zero-knowledge in
  the FRI query phase (high-end padding + `alpha*f + f'` combination threaded through
  the query/consistency path) remains the final A2 step and is not claimed here.

- `crates/core/src/piop/mod.rs`
  Register `mod zk;` and re-export `augment_commit_meta_with_mask`, `mask_multilinear`,
  `mask_n_vars`.

- `crates/core/src/protocols/fri/zk_pcs.rs` (new file)
  A2 final query-phase piece (technique 2 of Construction 4.1): `prove_combination` /
  `verify_combination`, a two-tree FRI prover/verifier that runs the low-degree test on
  the virtual combination `alpha*f + f'`. `f` (the technique-1-padded witness codeword)
  and `f'` (fresh random) are committed in separate Merkle trees whose roots are bound
  into the transcript BEFORE `alpha` is sampled; `f'` keeps the folded intermediate
  codewords hiding as the FRI domain shrinks. The design reuses the vetted, unchanged
  `FRIFolder`/`FRIVerifier` on the combined codeword (a linear combination of two
  codewords is itself a codeword — no re-encode) and adds a per-query cross-check that
  the combined coset equals `alpha*f_coset + f'_coset` against the two pre-alpha trees,
  binding `alpha` to the pre-commitments. Tests: honest round-trip verifies; an
  adversarial wrong-`alpha` proof is rejected. Opt-in; changes no existing code path.

- `crates/core/src/protocols/fri/mod.rs`
  Register `mod zk_pcs;` and re-export `prove_combination`, `verify_combination`.

- `crates/core/src/protocols/sumcheck/zk.rs` (new file)
  A3 (zero-knowledge roadmap) — closes the "higher-level PIOP" caveat left open by Diamond's ZK PCS:
  the constraint-system sumcheck reveals trace evaluations and must be masked too. `MaskCombine`
  is the Libra/Chiesa–Forbes–Spooner masking composition `(f, g) -> f + rho*g`; running the standard
  sumcheck with it over `[f, g]` (fresh random mask `g`, its sum `s` revealed, `rho` a verifier
  challenge) is a zero-knowledge sumcheck for `f` — the emitted round polynomials belong to
  `f + rho*g` and are masked by `g`. The test drives binius's own `RegularSumcheckProver` /
  `batch_prove` / `batch_verify` end to end, confirming the masked sumcheck soundly reduces the
  `f + rho*g` claim and that `v = combined_sum - rho*s` recovers the base sum without revealing `f`'s
  round polynomials. Opt-in; wiring into `sumcheck::zerocheck` (the AIR constraint check) is the
  remaining integration step.

- `crates/core/src/protocols/sumcheck/mod.rs`
  Register `pub mod zk;`.

All other files are unmodified from upstream. This derivative work continues to be
licensed under the Apache License, Version 2.0.
