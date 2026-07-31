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

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField, TowerField};
use binius_ntt::AdditiveNTT;
use binius_utils::SerializeBytes;
use binius_math::MultilinearExtension;
use rand::Rng;

use super::{error::Error, verify::CommitMeta};
use crate::{
	fiat_shamir::{CanSample, Challenger},
	merkle_tree::{MerkleTreeProver, MerkleTreeScheme},
	protocols::fri::{
		self, prove_combination, verify_combination, CommitOutput, FRIParams,
	},
	transcript::{ProverTranscript, VerifierTranscript},
};

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

// ---------------------------------------------------------------------------------------------
// A2 two-tree ZK opening, wired onto a real piop commitment.
//
// binius's `piop::prove` interleaves its FRI with the batch sumcheck, so the pointwise two-tree
// combination `alpha*f + f'` (`fri::zk_pcs`) cannot be dropped into that interleaved loop without
// replacing the PCS. What DOES compose cleanly — and is what these helpers provide — is running the
// two-tree combination as a **zero-knowledge low-degree / opening proof on the piop-committed
// codeword** `f`: the caller commits its witness batch with `piop::commit` (optionally with the A2
// masking column for technique 1), and `prove_zk_opening` proves that committed codeword is close to
// the RS code while hiding its opened symbols behind a fresh random companion `f'` (technique 2).
// The evaluation-binding sumcheck is the separately-masked A3 path (`sumcheck::zk`); together they
// give an opened-value-ZK opening of the piop commitment. `alpha` is bound after both `f` and `f'`
// are committed, exactly as in `fri::zk_pcs`.

/// Proves, in zero-knowledge, that the piop-committed codeword `codeword_f` (commitment `root_f`,
/// Merkle state `committed_f`) is a low-degree codeword, hiding its opened symbols behind a fresh
/// random companion `f'` built from the caller-supplied random message `fprime_message` (a message of
/// the same shape the piop commit consumes; the caller supplies the randomness, so this carries no
/// RNG dependency). Writes `root_f`, the companion commitment, `alpha`-derived material, and the
/// two-tree query openings to `transcript`.
#[allow(clippy::too_many_arguments)]
pub fn prove_zk_opening<F, FA, P, NTT, MTProver, VCS, Challenger_>(
	fri_params: &FRIParams<F, FA>,
	ntt: &NTT,
	merkle_prover: &MTProver,
	codeword_f: &[P],
	committed_f: &MTProver::Committed,
	root_f: &VCS::Digest,
	fprime_message: &[P],
	transcript: &mut ProverTranscript<Challenger_>,
) -> Result<(), Error>
where
	F: TowerField + ExtensionField<FA>,
	FA: BinaryField,
	P: PackedField<Scalar = F> + PackedExtension<FA>,
	NTT: AdditiveNTT<FA> + Sync,
	MTProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F, Digest: SerializeBytes>,
	Challenger_: Challenger,
{
	// Commit the fresh random companion f' with the same FRI parameters as f.
	let CommitOutput { commitment: root_fprime, committed: committed_fprime, codeword: codeword_fprime } =
		fri::commit_interleaved(fri_params.rs_code(), fri_params, ntt, merkle_prover, fprime_message)?;

	// Bind alpha to BOTH commitments: write root_f and root_f' before sampling.
	transcript.message().write(root_f);
	transcript.message().write(&root_fprime);
	let alpha: F = transcript.sample();

	prove_combination(
		fri_params,
		ntt,
		merkle_prover,
		codeword_f,
		committed_f,
		&codeword_fprime,
		&committed_fprime,
		alpha,
		transcript,
	)?;
	Ok(())
}

/// Verifies a [`prove_zk_opening`] proof against the known piop commitment `root_f`. Returns the
/// fully-folded value of the combined codeword.
pub fn verify_zk_opening<F, FA, VCS, Challenger_>(
	fri_params: &FRIParams<F, FA>,
	vcs: &VCS,
	root_f: &VCS::Digest,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<F, Error>
where
	F: TowerField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: MerkleTreeScheme<F, Digest: binius_utils::DeserializeBytes + PartialEq>,
	Challenger_: Challenger,
{
	// Read the two commitments the prover bound; check the first is the claimed piop commitment.
	let root_f_read: VCS::Digest =
		transcript.message().read().map_err(|e| Error::FRI(fri::Error::TranscriptError(e)))?;
	if &root_f_read != root_f {
		return Err(Error::FRI(fri::Error::InvalidArgs(
			"zk opening: committed codeword root does not match the claimed commitment".into(),
		)));
	}
	let root_fprime: VCS::Digest =
		transcript.message().read().map_err(|e| Error::FRI(fri::Error::TranscriptError(e)))?;
	let alpha: F = transcript.sample();

	let final_value = verify_combination(fri_params, vcs, &root_f_read, &root_fprime, alpha, transcript)?;
	Ok(final_value)
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
	fn two_tree_zk_opening_on_a_real_piop_commitment() {
		use super::{prove_zk_opening, verify_zk_opening};
		use crate::{fiat_shamir::CanSample, protocols::fri};

		let merkle_prover =
			BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);
		let merkle_scheme = merkle_prover.scheme();
		let mut rng = StdRng::seed_from_u64(9);

		// A real committed batch, committed exactly as the piop does.
		let n_vars = 8usize;
		let commit_meta = CommitMeta::with_vars([n_vars]);
		let fri_params = make_commit_params_with_optimal_arity::<_, FEncode, _>(
			&commit_meta, merkle_scheme, SECURITY_BITS, LOG_INV_RATE,
		)
		.unwrap();
		let ntt = SingleThreadedNTT::new(fri_params.rs_code().log_len()).unwrap();

		let real_len = 1 << n_vars.saturating_sub(P::LOG_WIDTH);
		let real: MultilinearExtension<P> = MultilinearExtension::new(
			n_vars,
			std::iter::repeat_with(|| P::random(&mut rng)).take(real_len).collect(),
		)
		.unwrap();
		let committed_multilins = vec![MLEDirectAdapter::from(real)];
		let CommitOutput { commitment: root_f, committed: committed_f, codeword: codeword_f } =
			commit(&fri_params, &ntt, &merkle_prover, &committed_multilins).unwrap();

		// Fresh random companion message f' of the same shape (caller-supplied randomness).
		let fprime_message: Vec<P> = std::iter::repeat_with(|| P::random(&mut rng))
			.take((fri_params.rs_code().dim() << fri_params.log_batch_size()) >> P::LOG_WIDTH)
			.collect();

		// Prove a zero-knowledge two-tree opening of the piop-committed codeword.
		let mut proof = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		prove_zk_opening(
			&fri_params, &ntt, &merkle_prover, &codeword_f, &committed_f, &root_f, &fprime_message,
			&mut proof,
		)
		.unwrap();
		let prover_sample = CanSample::<F>::sample(&mut proof);

		// Verify against the known piop commitment root_f.
		let mut proof = proof.into_verifier();
		verify_zk_opening(&fri_params, merkle_scheme, &root_f, &mut proof)
			.expect("zk opening of the piop commitment must verify");
		assert_eq!(prover_sample, CanSample::<F>::sample(&mut proof), "transcripts stay in sync");

		// A verifier that presents the WRONG commitment must be rejected (the opening is bound to
		// the actual piop commitment).
		let mut proof2 = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		prove_zk_opening(
			&fri_params, &ntt, &merkle_prover, &codeword_f, &committed_f, &root_f, &fprime_message,
			&mut proof2,
		)
		.unwrap();
		let mut proof2 = proof2.into_verifier();
		let wrong_root = fri::commit_interleaved(
			fri_params.rs_code(),
			&fri_params,
			&ntt,
			&merkle_prover,
			&fprime_message,
		)
		.unwrap()
		.commitment;
		assert!(
			verify_zk_opening(&fri_params, merkle_scheme, &wrong_root, &mut proof2).is_err(),
			"an opening presented against the wrong commitment must be rejected"
		);
	}

	#[test]
	fn mask_lands_in_its_own_largest_bucket() {
		let meta = CommitMeta::with_vars([6, 7, 7]);
		assert_eq!(mask_n_vars::<F>(&meta), 8);
		let augmented = augment_commit_meta_with_mask::<F>(&meta);
		assert_eq!(augmented.n_multilins_by_vars(), &[0, 0, 0, 0, 0, 0, 1, 2, 1]);
	}
}
