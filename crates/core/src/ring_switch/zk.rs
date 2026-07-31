// Copyright 2026 The pq-rollup Authors
// Addition to the Binius fork (Apache-2.0). See MODIFICATIONS.md.
//
// A-final (zero-knowledge roadmap) — closing the ring-switching caveat. Diamond's ZK Binius PCS
// (IACR ePrint 2025/1015) makes the *large-field* commitment zero-knowledge but explicitly leaves
// the *ring-switching* reduction (Diamond–Posen 2024/504, §4) outside the ZK boundary. Our AIRs are
// over B1, and ring-switching is what reduces a small-field (B1) evaluation claim to a large-field
// sumcheck, so this gap was on our path.
//
// Closure (statistical, quantum-safe by default — as predicted in paper/zk-qrom-proof.md §7):
//
//   The ring-switching prover (`ring_switch::prove`) sends exactly two witness-dependent things to
//   the transcript: (1) the batched *partial evaluations* of the committed multilinears — tensor-
//   algebra elements `mixed_tensor_elem.vertical_elems()` (prove.rs, ~l.87), computed by
//   `compute_partial_evals` = a partial multilinear evaluation (fold-high) of each witness; and
//   (2) the `row_batched_evals` `s'` (prove.rs, ~l.98). For FIXED Fiat–Shamir challenges (the mixing
//   and row-batching randomness), BOTH are **F-linear maps of the witness**: partial multilinear
//   evaluation is F-linear, and scaling by mixing coefficients, mixing across prefixes, and
//   row-batching are all fixed F-linear combinations of those partial evaluations.
//
//   Therefore masking the committed witness `t ↦ t + m` — which the upstream ZK layer ALREADY does
//   (A2 high-end padding / masking column, A3 mask polynomial), so ring-switching sees `t + m` — makes
//   every ring-switch emission `emit(t) + emit(m)`. With `m` uniform over the large field `L`, the
//   emissions are uniformly masked: statistical hiding with error `≤ 2^{-log2|L|} = 2^{-field_bits}`.
//   No random oracle enters, so this is identical in the ROM and the QROM — ring-switching inherits
//   zero-knowledge from the commitment-level mask, quantum-safe by default, and needs NO separate
//   oracle-programming step. The reduced sum `s'` it emits is exactly the sumcheck claim consumed by
//   the (A3-masked) downstream sumcheck, so it introduces no leakage beyond that already-ZK step.
//
// This module documents the closure and provides the test that validates its load-bearing fact —
// partial multilinear evaluation (the "MLE Fold High" at the heart of `compute_partial_evals`) is
// F-linear — against binius's own `evaluate_partial_high`. Everything downstream (scaling, mixing,
// row-batching) is a manifest fixed linear combination, so witness-linearity of the whole reduction
// follows.

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField128b, Field, PackedField};
	use binius_math::{MultilinearExtension, MultilinearQuery};
	use rand::{rngs::StdRng, SeedableRng};

	type F = BinaryField128b;

	fn rand_mle(rng: &mut StdRng, n_vars: usize) -> MultilinearExtension<F> {
		let evals: Vec<F> = (0..(1 << n_vars)).map(|_| <F as Field>::random(&mut *rng)).collect();
		MultilinearExtension::from_values(evals).unwrap()
	}

	/// The ring-switching reduction's witness-dependent emissions are F-linear because their core
	/// operation — partial multilinear evaluation (fold-high) — is F-linear. This validates that
	/// against binius's real `evaluate_partial_high`, so masking the committed witness masks the
	/// ring-switch emissions (statistical ZK over the large field, no oracle).
	#[test]
	fn partial_evaluation_is_linear_so_masking_the_witness_masks_the_emissions() {
		let mut rng = StdRng::seed_from_u64(0x21_09_5217);
		let n_vars = 8;
		let k = 3; // number of high variables folded (the "kappa"-like split)

		let f = rand_mle(&mut rng, n_vars);
		let g = rand_mle(&mut rng, n_vars);
		let alpha = <F as Field>::random(&mut rng);

		// The masked witness as an MLE: (alpha*f + g), formed on the evaluation vectors.
		let combined_evals: Vec<F> = f
			.evals()
			.iter()
			.zip(g.evals())
			.map(|(&a, &b)| alpha * a + b)
			.collect();
		let combined = MultilinearExtension::<F>::from_values(combined_evals).unwrap();

		// Fold the high `k` variables at a random point (the ring-switch "MLE Fold High" step).
		let r: Vec<F> = (0..k).map(|_| <F as Field>::random(&mut rng)).collect();
		let query = MultilinearQuery::<F>::expand(&r);

		let pf = f.evaluate_partial_high(query.to_ref()).unwrap();
		let pg = g.evaluate_partial_high(query.to_ref()).unwrap();
		let pc = combined.evaluate_partial_high(query.to_ref()).unwrap();

		// partial(alpha*f + g) == alpha*partial(f) + partial(g), coefficient by coefficient.
		let expected: Vec<F> = pf
			.evals()
			.iter()
			.zip(pg.evals())
			.map(|(&a, &b)| alpha * a + b)
			.collect();
		let got: Vec<F> = PackedField::iter_slice(pc.evals()).take(1 << (n_vars - k)).collect();
		assert_eq!(got, expected, "ring-switch fold-high is F-linear in the witness");

		// Mirror of the ZK argument's scaling half: folding (alpha*f) equals alpha*folding(f), so a
		// scaled mask scales its emission — the mask stays live and independent of f's structure.
		let scaled_evals: Vec<F> = f.evals().iter().map(|&a| alpha * a).collect();
		let scaled = MultilinearExtension::<F>::from_values(scaled_evals).unwrap();
		let scaled_fold: Vec<F> =
			PackedField::iter_slice(scaled.evaluate_partial_high(query.to_ref()).unwrap().evals())
				.take(1 << (n_vars - k))
				.collect();
		let alpha_pf: Vec<F> = pf.evals().iter().map(|&a| alpha * a).collect();
		assert_eq!(scaled_fold, alpha_pf);
	}
}
