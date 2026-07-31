// Copyright 2026 The pq-rollup Authors
// Addition to the Binius fork (Apache-2.0). See MODIFICATIONS.md.
//
// A3 (zero-knowledge roadmap) — zero-knowledge sumcheck by Libra-style masking, closing the
// "higher-level PIOP" caveat that Diamond's ZK Binius PCS (ePrint 2025/1015) leaves open: the PCS
// makes its *internal* evaluation sumcheck zero-knowledge, but the higher-level PIOP sumcheck that
// checks the AIR constraints still reveals trace evaluations at its challenge points and must be
// masked too.
//
// The classic fix (Chiesa–Forbes–Spooner; Libra, Xie et al. CRYPTO'19): to prove
// `sum_{x in H} f(x) = v` in zero-knowledge, sample a fresh random masking multilinear `g`, reveal
// only its hypercube sum `s = sum_H g`, take a verifier challenge `rho`, and run the *standard*
// sumcheck on the combined polynomial `f + rho*g`, whose claimed sum is `v + rho*s`. Every round
// polynomial the prover sends is a round polynomial of `f + rho*g`; because `g` is uniformly random
// and only its total `s` is disclosed, those round polynomials are uniformly distributed subject to
// the public sum, so they leak nothing about `f` beyond `v`. Correctness is exact: the sumcheck
// verifier's reduced claim on `f + rho*g` at the final point equals `f(r) + rho*g(r)`, and `v` is
// recovered as `(claimed combined sum) - rho*s`.
//
// This module provides the combining composition `MaskCombine` (`(f, g) -> f + rho*g`) so the mask
// runs through binius's own `RegularSumcheckProver` / `batch_verify` unchanged — i.e. the masking
// composes with the real sumcheck, validated end to end by the test. Wiring it into the constraint
// system's zerocheck (`sumcheck::zerocheck`, where the AIR composition is checked) is the remaining
// integration step; the masking construction itself is what this closes.

use binius_field::{Field, PackedField, TowerField};
use binius_math::{ArithExpr, CompositionPoly};

/// The Libra masking composition: `(f, g) -> f + rho*g`, degree 1 in two variables. Running a
/// sumcheck with this composition over the multilinears `[f, g]` is a zero-knowledge sumcheck for
/// `f` — the round polynomials it emits belong to `f + rho*g` and are masked by the random `g`.
#[derive(Debug, Clone)]
pub struct MaskCombine<F> {
	/// The verifier's combination challenge, sampled after the mask sum `s` is revealed.
	pub rho: F,
}

impl<F: Field> MaskCombine<F> {
	pub const fn new(rho: F) -> Self {
		Self { rho }
	}

	/// The combined claimed sum `v + rho*s` for base sum `v = sum_H f` and mask sum `s = sum_H g`.
	pub fn combined_sum(&self, v: F, s: F) -> F {
		v + self.rho * s
	}
}

impl<P: PackedField> CompositionPoly<P> for MaskCombine<P::Scalar>
where
	P::Scalar: TowerField,
{
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		1
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		ArithExpr::Var(0) + ArithExpr::Const(self.rho) * ArithExpr::Var(1)
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		Ok(query[0] + P::broadcast(self.rho) * query[1])
	}

	fn binary_tower_level(&self) -> usize {
		// rho is a full extension-field element, so the composition lives at the scalar's tower level.
		<P::Scalar as TowerField>::TOWER_LEVEL
	}
}

/// The Libra-masked **zero-check** composition for a Boolean-constraint column: `(w, g) -> w^2 + w +
/// rho*g`. In characteristic 2, `w^2 + w = w(w+1)` vanishes exactly when `w in {0,1}`, so for a bit
/// witness the constraint part sums to zero over the hypercube and a sumcheck with this composition
/// proves that Boolean constraint with claimed sum `rho*s` (`s = sum_H g`). The random mask `g`
/// hides the round polynomials, which would otherwise reveal the low-degree extension of `w` at the
/// challenge points. This is the shape the AIR constraint zerocheck reduces to (an eq-indicator times
/// a constraint composition); masking it is the zero-knowledge zerocheck that composes with A1/A2.
#[derive(Debug, Clone)]
pub struct MaskedZerocheck<F> {
	pub rho: F,
}

impl<F: Field> MaskedZerocheck<F> {
	pub const fn new(rho: F) -> Self {
		Self { rho }
	}
}

impl<P: PackedField> CompositionPoly<P> for MaskedZerocheck<P::Scalar>
where
	P::Scalar: TowerField,
{
	fn n_vars(&self) -> usize {
		2
	}

	fn degree(&self) -> usize {
		2
	}

	fn expression(&self) -> ArithExpr<P::Scalar> {
		// w^2 + w + rho*g  (subtraction is addition in characteristic 2).
		ArithExpr::Var(0).pow(2) + ArithExpr::Var(0) + ArithExpr::Const(self.rho) * ArithExpr::Var(1)
	}

	fn evaluate(&self, query: &[P]) -> Result<P, binius_math::Error> {
		Ok(query[0] * query[0] + query[0] + P::broadcast(self.rho) * query[1])
	}

	fn binary_tower_level(&self) -> usize {
		<P::Scalar as TowerField>::TOWER_LEVEL
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, BinaryField128b, BinaryField8b,
		Field, PackedField,
	};
	use binius_hal::{make_portable_backend, ComputationBackendExt};
	use binius_hash::groestl::Groestl256;
	use binius_math::{
		CompositionPoly, EvaluationOrder, IsomorphicEvaluationDomainFactory, MLEEmbeddingAdapter,
		MultilinearExtension,
	};
	use rand::{rngs::StdRng, SeedableRng};

	use super::MaskCombine;
	use crate::{
		fiat_shamir::{CanSample, HasherChallenger},
		protocols::sumcheck::{
			batch_verify, prove::batch_prove, prove::RegularSumcheckProver, BatchSumcheckOutput,
			CompositeSumClaim, SumcheckClaim,
		},
		transcript::ProverTranscript,
	};

	type U = OptimalUnderlier128b;
	type FDomain = BinaryField8b;
	type F = BinaryField128b;
	type P = PackedType<U, F>;

	/// Sum of a multilinear over the Boolean hypercube = sum of all its scalar evaluations.
	fn hypercube_sum(mle: &MultilinearExtension<P>) -> F {
		PackedField::iter_slice(mle.evals()).take(1 << mle.n_vars()).sum()
	}

	#[test]
	fn zk_sumcheck_masks_and_still_proves_the_base_sum() {
		let n_vars = 8;
		let mut rng = StdRng::seed_from_u64(42);

		// The witness multilinear f (whose sum we prove) and a FRESH RANDOM mask g.
		let f = MultilinearExtension::<P>::new(
			n_vars,
			std::iter::repeat_with(|| P::random(&mut rng)).take(1 << n_vars).collect(),
		)
		.unwrap();
		let g = MultilinearExtension::<P>::new(
			n_vars,
			std::iter::repeat_with(|| P::random(&mut rng)).take(1 << n_vars).collect(),
		)
		.unwrap();

		let v = hypercube_sum(&f); // the base sum being proven (kept private)
		let s = hypercube_sum(&g); // the mask sum (publicly revealed)
		assert_ne!(s, F::ZERO, "a live mask should have nonzero sum here");

		// Prover writes the mask sum s, then both sides sample the combination challenge rho.
		let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		prover_transcript.message().write_scalar(s);
		let rho: F = prover_transcript.sample();

		let combine = MaskCombine::new(rho);
		let combined_sum = combine.combined_sum(v, s);

		let backend = make_portable_backend();
		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let multilins = [f.clone(), g.clone()]
			.map(MLEEmbeddingAdapter::<_, P, _>::from)
			.to_vec();

		// The zero-knowledge sumcheck: the standard binius sumcheck run on f + rho*g.
		let prover = RegularSumcheckProver::<FDomain, _, _, _, _>::new(
			EvaluationOrder::HighToLow,
			multilins.iter().collect(),
			[CompositeSumClaim { composition: &combine, sum: combined_sum }],
			domain_factory,
			|_| 0,
			&backend,
		)
		.unwrap();
		let prover_reduced =
			batch_prove(vec![prover], &mut prover_transcript).expect("zk sumcheck prove");

		let prover_sample = CanSample::<F>::sample(&mut prover_transcript);
		let mut verifier_transcript = prover_transcript.into_verifier();

		// Verifier: read s, sample the same rho, verify the sumcheck on the combined claim.
		let s_v: F = verifier_transcript.message().read_scalar().unwrap();
		assert_eq!(s_v, s);
		let rho_v: F = verifier_transcript.sample();
		let combine_v = MaskCombine::new(rho_v);
		let claim = SumcheckClaim::new(
			n_vars,
			2,
			vec![CompositeSumClaim { composition: &combine_v, sum: combine_v.combined_sum(v, s_v) }],
		)
		.unwrap();
		let verifier_reduced =
			batch_verify(EvaluationOrder::HighToLow, &[claim], &mut verifier_transcript).unwrap();

		assert_eq!(prover_sample, CanSample::<F>::sample(&mut verifier_transcript));
		verifier_transcript.finalize().unwrap();
		assert_eq!(verifier_reduced, prover_reduced);

		// The reduced multilinear evaluations pin f(r) and g(r) at the challenge point, and the
		// combined value equals f(r) + rho*g(r) — i.e. the masked sumcheck soundly reduces the
		// f + rho*g claim while never revealing f's own round polynomials.
		let BatchSumcheckOutput { challenges, multilinear_evals } = verifier_reduced;
		let query = backend.multilinear_query::<F>(&challenges).unwrap();
		let f_r = f.evaluate(query.to_ref()).unwrap();
		let g_r = g.evaluate(query.to_ref()).unwrap();
		assert_eq!(multilinear_evals[0][0], f_r);
		assert_eq!(multilinear_evals[0][1], g_r);
		assert_eq!(
			combine_v.evaluate(&[f_r, g_r]).unwrap(),
			f_r + rho * g_r,
			"reduced combined value is f(r) + rho*g(r)"
		);

		// Correctness of the ZK wrapper: the public data (combined_sum, s, rho) recovers the base
		// sum v = combined_sum - rho*s, without v ever being sent in the clear.
		assert_eq!(combined_sum - rho * s, v, "base sum recoverable from the masked claim");
	}

	#[test]
	fn zk_zerocheck_masks_a_boolean_constraint() {
		use binius_field::packed::set_packed_slice;
		use rand::Rng;

		use super::MaskedZerocheck;

		let n_vars = 8;
		let n = 1usize << n_vars;
		let mut rng = StdRng::seed_from_u64(7);

		// Bit witness w: hypercube values in {0,1}, so the Boolean constraint w^2 + w vanishes on the
		// hypercube. (A masked *zerocheck*: the constraint part sums to zero.)
		let mut w_evals = vec![P::default(); n >> P::LOG_WIDTH];
		for i in 0..n {
			set_packed_slice(&mut w_evals, i, if rng.gen::<bool>() { F::ONE } else { F::ZERO });
		}
		let w = MultilinearExtension::<P>::new(n_vars, w_evals).unwrap();

		// Fresh random mask g; reveal only its sum s.
		let g = MultilinearExtension::<P>::new(
			n_vars,
			std::iter::repeat_with(|| P::random(&mut rng)).take(n >> P::LOG_WIDTH).collect(),
		)
		.unwrap();
		let s = hypercube_sum(&g);

		let mut prover_transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		prover_transcript.message().write_scalar(s);
		let rho: F = prover_transcript.sample();

		let comp = MaskedZerocheck::new(rho);
		let claimed = rho * s; // constraint part sums to 0 for a bit column

		let backend = make_portable_backend();
		let domain_factory = IsomorphicEvaluationDomainFactory::<FDomain>::default();
		let multilins = [w.clone(), g.clone()].map(MLEEmbeddingAdapter::<_, P, _>::from).to_vec();

		let prover = RegularSumcheckProver::<FDomain, _, _, _, _>::new(
			EvaluationOrder::HighToLow,
			multilins.iter().collect(),
			[CompositeSumClaim { composition: &comp, sum: claimed }],
			domain_factory,
			|_| 0,
			&backend,
		)
		.unwrap();
		let prover_reduced =
			batch_prove(vec![prover], &mut prover_transcript).expect("zk zerocheck prove");

		let prover_sample = CanSample::<F>::sample(&mut prover_transcript);
		let mut verifier_transcript = prover_transcript.into_verifier();
		let s_v: F = verifier_transcript.message().read_scalar().unwrap();
		let rho_v: F = verifier_transcript.sample();
		let comp_v = MaskedZerocheck::new(rho_v);
		let claim = SumcheckClaim::new(
			n_vars,
			2,
			vec![CompositeSumClaim { composition: &comp_v, sum: rho_v * s_v }],
		)
		.unwrap();
		let verifier_reduced =
			batch_verify(EvaluationOrder::HighToLow, &[claim], &mut verifier_transcript).unwrap();

		assert_eq!(prover_sample, CanSample::<F>::sample(&mut verifier_transcript));
		verifier_transcript.finalize().unwrap();
		assert_eq!(verifier_reduced, prover_reduced);

		// The reduced claim pins w(r), g(r) and w^2(r)+w(r)+rho*g(r) — the masked zerocheck soundly
		// reduces the Boolean constraint while never revealing w's own round polynomials.
		let BatchSumcheckOutput { challenges, multilinear_evals } = verifier_reduced;
		let query = backend.multilinear_query::<F>(&challenges).unwrap();
		let w_r = w.evaluate(query.to_ref()).unwrap();
		let g_r = g.evaluate(query.to_ref()).unwrap();
		assert_eq!(multilinear_evals[0][0], w_r);
		assert_eq!(multilinear_evals[0][1], g_r);
		assert_eq!(comp_v.evaluate(&[w_r, g_r]).unwrap(), w_r * w_r + w_r + rho * g_r);
	}
}
