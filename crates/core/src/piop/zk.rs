// Copyright 2026 The pq-rollup Authors
// Addition to the Binius fork (Apache-2.0). See MODIFICATIONS.md.
//
// A2 (zero-knowledge roadmap) — protocol-level wiring of masking into the FRI-Binius PCS
// (`crate::piop`). This is the step that carries the zero-knowledge helpers of
// `protocols::fri::zk` (ported from Diamond, IACR ePrint 2025/1015) up into the interleaved
// sumcheck-FRI commit/prove/verify compiler.
//
// The commit-level construction here is deliberately the *protocol-compatible* one: it appends a
// freshly-sampled masking multilinear to the committed batch, in its own (largest) variable bucket.
// Because that bucket carries no sumcheck claim it is an unconstrained committed column — a shape
// the existing `prove`/`verify` already support (see `piop::tests::test_without_opening_claims`) —
// so NO change to `prove`, `verify`, `FRIParams`, or the verifier consistency check is required.
// The mask's coefficients enter the merged FRI message, so the commitment and the codeword are
// randomised: the commitment is hiding, and two commitments of the same batch under independent
// masks are unlinkable, while every real opening still verifies.
//
// Scope boundary (honest): this realises *commitment* hiding (the A4 "masking column" technique at
// PCS scope). Full zero-knowledge of the *opened values* in the FRI query phase additionally needs
// technique-1 high-end padding and the technique-2 `alpha*f + f'` low-degree-test combination
// threaded through the FRI query/consistency path (`protocols::fri::zk` provides those primitives;
// wiring them requires changing `FRIParams` dimensioning and the verifier's final check). That
// remains the last A2 step; it is not claimed here.

use binius_field::{PackedField, TowerField};
use binius_math::MultilinearExtension;
use rand::Rng;

use super::verify::CommitMeta;

/// Samples a fresh, fully-random masking multilinear on `n_vars` variables. Committing this
/// alongside the real batch randomises the merged FRI message, making the commitment hiding.
pub fn mask_multilinear<P>(n_vars: usize, mut rng: impl Rng) -> MultilinearExtension<P>
where
	P: PackedField,
{
	let len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
	let evals = std::iter::repeat_with(|| P::random(&mut rng)).take(len).collect::<Vec<_>>();
	MultilinearExtension::new(n_vars, evals).expect("len matches n_vars")
}

/// The number of variables to give the masking multilinear: one above the batch's current maximum,
/// so the mask lands in its own fresh bucket (guaranteed unconstrained, and strictly largest, so the
/// committed batch stays sorted ascending when the mask is appended last).
pub fn mask_n_vars<F: TowerField>(commit_meta: &CommitMeta) -> usize {
	commit_meta.max_n_vars() + 1
}

/// Returns the commit metadata for the zero-knowledge batch: the input batch plus one masking
/// multilinear at [`mask_n_vars`]. The prover appends [`mask_multilinear`] (with these `n_vars`) to
/// its committed multilinears; both prover and verifier derive their `FRIParams` from this augmented
/// metadata. The original sumcheck claims are unchanged — the mask is referenced by none of them.
pub fn augment_commit_meta_with_mask<F: TowerField>(commit_meta: &CommitMeta) -> CommitMeta {
	let mask_vars = mask_n_vars::<F>(commit_meta);
	let mut counts = commit_meta.n_multilins_by_vars().to_vec();
	if mask_vars >= counts.len() {
		counts.resize(mask_vars + 1, 0);
	}
	counts[mask_vars] += 1;
	CommitMeta::new(counts)
}

#[cfg(test)]
mod tests {
	use binius_field::{
		BinaryField128b, BinaryField16b, BinaryField8b, PackedBinaryField2x128b, PackedField,
	};
	use binius_hal::make_portable_backend;
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
	use binius_math::{
		DefaultEvaluationDomainFactory, MLEDirectAdapter, MultilinearExtension, MultilinearPoly,
	};
	use binius_ntt::SingleThreadedNTT;
	use rand::{rngs::StdRng, SeedableRng};

	use super::{augment_commit_meta_with_mask, mask_multilinear, mask_n_vars};
	use crate::{
		fiat_shamir::HasherChallenger,
		merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme},
		piop::{
			prove,
			prove::commit,
			verify,
			verify::{make_commit_params_with_optimal_arity, CommitMeta},
			PIOPSumcheckClaim,
		},
		polynomial::MultivariatePoly,
		protocols::fri::CommitOutput,
		transcript::ProverTranscript,
		transparent,
	};

	type F = BinaryField128b;
	type FDomain = BinaryField8b;
	type FEncode = BinaryField16b;
	type P = PackedBinaryField2x128b;

	const SECURITY_BITS: usize = 32;
	const LOG_INV_RATE: usize = 1;

	/// Commit a single committed multilinear on `n_vars` variables, optionally with a masking
	/// column, run the full sumcheck-FRI prove/verify, and return the commitment digest so callers
	/// can compare commitments. Panics if verification fails.
	fn commit_prove_verify_zk(
		n_vars: usize,
		mask: bool,
		seed: u64,
	) -> <<BinaryMerkleTreeProver<F, Groestl256, Groestl256ByteCompression> as MerkleTreeProver<
		F,
	>>::Scheme as MerkleTreeScheme<F>>::Digest {
		let merkle_prover =
			BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
		let merkle_scheme = merkle_prover.scheme();
		let backend = make_portable_backend();
		let mut rng = StdRng::seed_from_u64(seed);

		// Base batch: one committed multilinear at `n_vars`.
		let base_meta = CommitMeta::with_vars([n_vars]);

		// The real committed multilinear (seed-independent, so masking is the only difference).
		let mut poly_rng = StdRng::seed_from_u64(0xC0DE);
		let real_len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
		let real: MultilinearExtension<P> = MultilinearExtension::new(
			n_vars,
			std::iter::repeat_with(|| P::random(&mut poly_rng)).take(real_len).collect(),
		)
		.unwrap();

		// Build the committed batch and its metadata, appending a mask column when requested.
		let (commit_meta, committed_multilins) = if mask {
			let mvars = mask_n_vars::<F>(&base_meta);
			let mask_mle = mask_multilinear::<P>(mvars, &mut rng);
			let committed = vec![
				MLEDirectAdapter::from(real.clone()),
				MLEDirectAdapter::from(mask_mle),
			];
			(augment_commit_meta_with_mask::<F>(&base_meta), committed)
		} else {
			(CommitMeta::with_vars([n_vars]), vec![MLEDirectAdapter::from(real.clone())])
		};

		let fri_params = make_commit_params_with_optimal_arity::<_, FEncode, _>(
			&commit_meta,
			merkle_scheme,
			SECURITY_BITS,
			LOG_INV_RATE,
		)
		.unwrap();
		let ntt = SingleThreadedNTT::new(fri_params.rs_code().log_len()).unwrap();

		let CommitOutput { commitment, committed, codeword } =
			commit(&fri_params, &ntt, &merkle_prover, &committed_multilins).unwrap();

		// One transparent + one sumcheck claim, on the REAL committed multilinear only (index 0).
		// The mask column (index 1) is referenced by no claim — it is unconstrained.
		let transparent_mle: MultilinearExtension<P> = MultilinearExtension::new(
			n_vars,
			std::iter::repeat_with(|| P::random(&mut poly_rng)).take(real_len).collect(),
		)
		.unwrap();
		let transparent_multilins = vec![MLEDirectAdapter::from(transparent_mle.clone())];

		let sum = (0..1 << n_vars)
			.map(|v| {
				committed_multilins[0].evaluate_on_hypercube(v).unwrap()
					* transparent_multilins[0].evaluate_on_hypercube(v).unwrap()
			})
			.sum();
		let sumcheck_claims =
			vec![PIOPSumcheckClaim { n_vars, committed: 0, transparent: 0, sum }];

		let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		proof.message().write(&commitment);

		let domain_factory = DefaultEvaluationDomainFactory::<FDomain>::default();
		prove(
			&fri_params,
			&ntt,
			&merkle_prover,
			domain_factory,
			&commit_meta,
			committed,
			&codeword,
			&committed_multilins,
			&transparent_multilins,
			&sumcheck_claims,
			&mut proof,
			&backend,
		)
		.unwrap();

		let mut proof = proof.into_verifier();

		let transparent_poly = transparent::MultilinearExtensionTransparent::<P, P>::from_values_and_mu(
			transparent_mle.evals().to_vec(),
			n_vars,
		)
		.unwrap();
		let transparent_polys: Vec<&dyn MultivariatePoly<F>> = vec![&transparent_poly];

		let read_commitment = proof.message().read().unwrap();
		verify(
			&commit_meta,
			merkle_scheme,
			&fri_params,
			&read_commitment,
			&transparent_polys,
			&sumcheck_claims,
			&mut proof,
		)
		.unwrap();

		commitment
	}

	#[test]
	fn zk_masked_commit_prove_verify_succeeds() {
		// The masked batch commits, proves, and verifies end-to-end through the unchanged
		// FRI-Binius prove/verify — the masking column is a valid unconstrained committed column.
		let _ = commit_prove_verify_zk(7, true, 1);
	}

	#[test]
	fn zk_mask_hides_the_commitment() {
		// Same real polynomial, two independent masks -> two different commitments (hiding), each of
		// which still proves and verifies. The unmasked commitment differs from both.
		let c_no_mask = commit_prove_verify_zk(7, false, 0);
		let c_mask_a = commit_prove_verify_zk(7, true, 1);
		let c_mask_b = commit_prove_verify_zk(7, true, 2);

		assert_ne!(c_mask_a, c_mask_b, "independent masks must give unlinkable commitments");
		assert_ne!(c_no_mask, c_mask_a, "masking must change the commitment");
	}

	#[test]
	fn mask_lands_in_its_own_largest_bucket() {
		let meta = CommitMeta::with_vars([6, 7, 7]);
		assert_eq!(mask_n_vars::<F>(&meta), 8);
		let augmented = augment_commit_meta_with_mask::<F>(&meta);
		assert_eq!(augmented.n_multilins_by_vars(), &[0, 0, 0, 0, 0, 0, 1, 2, 1]);
	}
}
