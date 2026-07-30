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

All other files are unmodified from upstream. This derivative work continues to be
licensed under the Apache License, Version 2.0.
