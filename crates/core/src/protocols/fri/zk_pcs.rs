// Copyright 2026 The pq-rollup Authors
// Addition to the Binius fork (Apache-2.0). See MODIFICATIONS.md.
//
// A2 (zero-knowledge roadmap) — the two-tree FRI prover/verifier for technique 2 of Diamond's
// zero-knowledge Binius commitment (IACR ePrint 2025/1015, Construction 4.1). This is the final
// query-phase piece of opened-value zero-knowledge.
//
// Technique 2 runs the FRI low-degree test on the virtual combination `f_combined = alpha*f + f'`,
// where `f` is the (technique-1 high-end-padded) committed witness codeword, `f'` is a fresh fully
// random codeword of the same length, and `alpha` is a verifier challenge sampled AFTER both `f`
// and `f'` are committed. The random `f'` keeps the *folded* intermediate codewords hiding even as
// the FRI domain shrinks (which high-end padding alone cannot do — the query count stays fixed while
// the domain contracts).
//
// Design (lowest soundness risk — reuses the vetted FRIFolder/FRIVerifier unchanged):
//   Prover:  commit f, f' separately (their roots go on the transcript BEFORE alpha is sampled);
//            form the combined codeword (a linear combination of two codewords is itself a codeword,
//            so no re-encode); commit it; run the standard FRIFolder on the combined codeword to
//            produce the round oracles + terminate codeword. Per query, additionally open the f and
//            f' cosets so the verifier can bind the combined coset to them.
//   Verifier: run the standard FRIVerifier low-degree/consistency checks on the combined codeword,
//            and per query additionally check `combined_coset == alpha*f_coset + f'_coset` against
//            the two pre-alpha trees. Because alpha is sampled after f and f' are committed and the
//            combined coset is bound to them at every queried position, a malicious prover cannot
//            commit a combined codeword unrelated to the pre-committed f, f'.
//
// Soundness rests entirely on: (a) the standard FRIVerifier (unchanged) for combined's proximity and
// fold consistency; (b) the per-query combination cross-check binding combined to the pre-alpha f,
// f' trees. Both are exercised by the tests, including an adversarial test that a wrong alpha (i.e.
// a combined codeword not bound to f, f' under the sampled alpha) is rejected.

use binius_field::{BinaryField, ExtensionField, PackedExtension, PackedField, TowerField};
use binius_ntt::{AdditiveNTT, SingleThreadedNTT};
use binius_utils::SerializeBytes;
use bytes::{Buf, BufMut};
use itertools::izip;

use super::{
	common::{vcs_optimal_layers_depths_iter, FRIParams},
	error::Error,
	prove::{to_par_scalar_big_chunks, to_par_scalar_small_chunks, FRIFolder, FoldRoundOutput},
	verify::FRIVerifier,
	VerificationError,
};
use crate::{
	fiat_shamir::{CanSample, CanSampleBits, Challenger},
	merkle_tree::{MerkleTreeProver, MerkleTreeScheme},
	transcript::{ProverTranscript, TranscriptReader, TranscriptWriter, VerifierTranscript},
};

/// Commits an already-encoded codeword with the same Merkle layout `commit_interleaved_with` uses,
/// so `FRIFolder`/`FRIVerifier` treat it identically. Returns the root and the committed state.
fn commit_codeword<F, FA, P, MTProver, VCS>(
	params: &FRIParams<F, FA>,
	merkle_prover: &MTProver,
	codeword: &[P],
) -> Result<(VCS::Digest, MTProver::Committed), Error>
where
	F: TowerField + ExtensionField<FA>,
	FA: BinaryField,
	P: PackedField<Scalar = F>,
	MTProver: MerkleTreeProver<F, Scheme = VCS>,
	VCS: MerkleTreeScheme<F>,
{
	let log_elems = params.rs_code().log_dim() + params.log_batch_size();
	let coset_log_len = params.fold_arities().first().copied().unwrap_or(log_elems);
	let log_len = params.log_len() - coset_log_len;

	let (commitment, committed) = if coset_log_len > P::LOG_WIDTH {
		merkle_prover
			.commit_iterated(to_par_scalar_big_chunks(codeword, 1 << coset_log_len), log_len)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?
	} else {
		merkle_prover
			.commit_iterated(to_par_scalar_small_chunks(codeword, 1 << coset_log_len), log_len)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?
	};
	Ok((commitment.root, committed))
}

/// Writes a coset opening: the `2^log_coset_size` coset values, then the Merkle path. Mirror of the
/// crate-private `prove_coset_opening`.
fn write_coset_opening<F, P, MTProver, B>(
	merkle_prover: &MTProver,
	codeword: &[P],
	committed: &MTProver::Committed,
	coset_index: usize,
	log_coset_size: usize,
	optimal_layer_depth: usize,
	advice: &mut TranscriptWriter<B>,
) -> Result<(), Error>
where
	F: TowerField,
	P: PackedField<Scalar = F>,
	MTProver: MerkleTreeProver<F>,
	B: BufMut,
{
	let values = binius_field::packed::iter_packed_slice_with_offset(
		codeword,
		coset_index << log_coset_size,
	)
	.take(1 << log_coset_size);
	advice.write_scalar_iter(values);
	merkle_prover
		.prove_opening(committed, optimal_layer_depth, coset_index, advice)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;
	Ok(())
}

/// Reads and verifies a coset opening, returning the coset values. Mirror of the crate-private
/// `verify_coset_opening`.
fn read_coset_opening<F, VCS, B>(
	vcs: &VCS,
	coset_index: usize,
	log_coset_size: usize,
	optimal_layer_depth: usize,
	tree_depth: usize,
	layer_digests: &[VCS::Digest],
	advice: &mut TranscriptReader<B>,
) -> Result<Vec<F>, Error>
where
	F: TowerField,
	VCS: MerkleTreeScheme<F>,
	B: Buf,
{
	let values = advice.read_scalar_slice::<F>(1 << log_coset_size)?;
	vcs.verify_opening(coset_index, &values, optimal_layer_depth, tree_depth, layer_digests, advice)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;
	Ok(values)
}

/// Proves the FRI low-degree test on the virtual combination `alpha*f + f'`.
///
/// `codeword_f` / `committed_f` and `codeword_fprime` / `committed_fprime` are the two codewords with
/// their Merkle trees, committed (their roots written to `transcript`) BEFORE `alpha` was sampled.
/// `alpha` is the combination challenge. Round oracles, the terminate codeword, and per-query
/// openings (combined, f, f') are written to `transcript`.
#[allow(clippy::too_many_arguments)]
pub fn prove_combination<F, FA, P, NTT, MTProver, VCS, Challenger_>(
	params: &FRIParams<F, FA>,
	ntt: &NTT,
	merkle_prover: &MTProver,
	codeword_f: &[P],
	committed_f: &MTProver::Committed,
	codeword_fprime: &[P],
	committed_fprime: &MTProver::Committed,
	alpha: F,
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
	// The combined codeword (linear combination of two codewords is a codeword; no re-encode).
	let alpha_bcast = P::broadcast(alpha);
	let combined: Vec<P> = codeword_f
		.iter()
		.zip(codeword_fprime)
		.map(|(&a, &b)| a * alpha_bcast + b)
		.collect();

	// Commit the combined codeword and write its root, then run the standard fold rounds.
	let (root_combined, committed_combined) = commit_codeword(params, merkle_prover, &combined)?;
	transcript.message().write(&root_combined);

	let mut folder = FRIFolder::new(params, ntt, merkle_prover, &combined, &committed_combined)?;
	for _ in 0..params.n_fold_rounds() {
		let challenge = transcript.sample();
		if let FoldRoundOutput::Commitment(round_commitment) = folder.execute_fold_round(challenge)? {
			transcript.message().write(&round_commitment);
		}
	}
	let (terminate_codeword, query_prover) = folder.finalize()?;

	// Header: terminate codeword, combined layers (base + rounds), then f and f' base layers.
	let first_depth = vcs_optimal_layers_depths_iter(params, merkle_prover.scheme())
		.next()
		.expect("at least the base oracle has an optimal layer depth");
	let f_layer = merkle_prover
		.layer(committed_f, first_depth)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?
		.to_vec();
	let fprime_layer = merkle_prover
		.layer(committed_fprime, first_depth)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?
		.to_vec();
	{
		let mut advice = transcript.decommitment();
		advice.write_scalar_slice(&terminate_codeword);
		for layer in query_prover.vcs_optimal_layers()? {
			advice.write_slice(&layer);
		}
		advice.write_slice(&f_layer);
		advice.write_slice(&fprime_layer);
	}

	// Per-query openings: combined (for the cross-check), f, f', then the standard combined chain.
	let first_arity = params.fold_arities().first().copied().unwrap_or(0);
	for _ in 0..params.n_test_queries() {
		let index = transcript.sample_bits(params.index_bits()) as usize;
		let mut advice = transcript.decommitment();
		write_coset_opening(
			merkle_prover, &combined, &committed_combined, index, first_arity, first_depth, &mut advice,
		)?;
		write_coset_opening(
			merkle_prover, codeword_f, committed_f, index, first_arity, first_depth, &mut advice,
		)?;
		write_coset_opening(
			merkle_prover, codeword_fprime, committed_fprime, index, first_arity, first_depth,
			&mut advice,
		)?;
		query_prover.prove_query(index, advice)?;
	}

	Ok(())
}

/// Verifies the two-tree FRI combination proof produced by [`prove_combination`].
///
/// `root_f` / `root_fprime` are the codeword commitments made before `alpha` was sampled; `alpha` is
/// the combination challenge. Returns the fully-folded value of the combined codeword.
#[allow(clippy::too_many_arguments)]
pub fn verify_combination<F, FA, VCS, Challenger_>(
	params: &FRIParams<F, FA>,
	vcs: &VCS,
	root_f: &VCS::Digest,
	root_fprime: &VCS::Digest,
	alpha: F,
	transcript: &mut VerifierTranscript<Challenger_>,
) -> Result<F, Error>
where
	F: TowerField + ExtensionField<FA>,
	FA: BinaryField,
	VCS: MerkleTreeScheme<F, Digest: binius_utils::DeserializeBytes>,
	Challenger_: Challenger,
{
	let ntt = SingleThreadedNTT::with_subspace(params.rs_code().subspace())?;

	// Read the combined-codeword root, then the interleaved fold-round challenges and round oracles,
	// mirroring the prover's Fiat–Shamir order.
	let root_combined: VCS::Digest = transcript.message().read().map_err(Error::TranscriptError)?;
	let mut challenges: Vec<F> = Vec::with_capacity(params.n_fold_rounds());
	let mut round_commitments: Vec<VCS::Digest> = Vec::with_capacity(params.n_oracles());
	for &arity in params.fold_arities().iter().take(params.n_oracles()) {
		let round_challenges: Vec<F> = transcript.sample_vec(arity);
		challenges.extend(round_challenges);
		let round_commitment: VCS::Digest =
			transcript.message().read().map_err(Error::TranscriptError)?;
		round_commitments.push(round_commitment);
	}
	let final_challenges: Vec<F> = transcript.sample_vec(params.n_final_challenges());
	challenges.extend(final_challenges);

	let verifier = FRIVerifier::new(params, vcs, &root_combined, &round_commitments, &challenges)?;

	// Header: terminate codeword + combined layers (base + rounds) + f and f' base layers. All are
	// kept alive because the per-query loop below needs the terminate slice and the layers.
	let terminate_len = 1 << (params.n_final_challenges() + params.rs_code().log_inv_rate());
	let first_depth = vcs_optimal_layers_depths_iter(params, vcs)
		.next()
		.expect("at least the base oracle has an optimal layer depth");

	let mut header = transcript.decommitment();
	let terminate_codeword: Vec<F> =
		header.read_scalar_slice(terminate_len).map_err(Error::TranscriptError)?;
	let final_value = verifier.verify_last_oracle(&ntt, &terminate_codeword)?;

	let combined_layers = vcs_optimal_layers_depths_iter(params, vcs)
		.map(|depth| header.read_vec(1 << depth))
		.collect::<Result<Vec<_>, _>>()
		.map_err(Error::TranscriptError)?;
	for (commitment, depth, layer) in izip!(
		std::iter::once(&root_combined).chain(&round_commitments),
		vcs_optimal_layers_depths_iter(params, vcs),
		&combined_layers
	) {
		vcs.verify_layer(commitment, depth, layer)
			.map_err(|err| Error::VectorCommit(Box::new(err)))?;
	}

	let f_layer = header.read_vec(1 << first_depth).map_err(Error::TranscriptError)?;
	vcs.verify_layer(root_f, first_depth, &f_layer)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;
	let fprime_layer = header.read_vec(1 << first_depth).map_err(Error::TranscriptError)?;
	vcs.verify_layer(root_fprime, first_depth, &fprime_layer)
		.map_err(|err| Error::VectorCommit(Box::new(err)))?;
	drop(header);

	// Per query: cross-check combined == alpha*f + f' against the two pre-alpha trees, then run the
	// standard combined-codeword consistency check via the unchanged FRIVerifier.
	let first_arity = params.fold_arities().first().copied().unwrap_or(0);
	for _ in 0..params.n_test_queries() {
		let index = transcript.sample_bits(params.index_bits()) as usize;
		let mut advice = transcript.decommitment();

		let c_vals = read_coset_opening(
			vcs, index, first_arity, first_depth, params.index_bits(), &combined_layers[0], &mut advice,
		)?;
		let f_vals = read_coset_opening(
			vcs, index, first_arity, first_depth, params.index_bits(), &f_layer, &mut advice,
		)?;
		let fprime_vals = read_coset_opening(
			vcs, index, first_arity, first_depth, params.index_bits(), &fprime_layer, &mut advice,
		)?;
		for ((&c, &f), &fp) in c_vals.iter().zip(&f_vals).zip(&fprime_vals) {
			if c != alpha * f + fp {
				return Err(VerificationError::IncorrectFold { query_round: 0, index }.into());
			}
		}
		verifier.verify_query(index, &ntt, &terminate_codeword, &combined_layers, &mut advice)?;
	}

	Ok(final_value)
}

#[cfg(test)]
mod tests {
	use std::iter::repeat_with;

	use binius_field::{
		arch::OptimalUnderlier128b, as_packed_field::PackedType, BinaryField128b, BinaryField16b,
		PackedField,
	};
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
	use binius_ntt::SingleThreadedNTT;
	use rand::{rngs::StdRng, SeedableRng};

	use super::{prove_combination, verify_combination};
	use crate::{
		fiat_shamir::{CanSample, HasherChallenger},
		merkle_tree::BinaryMerkleTreeProver,
		protocols::fri::{self, CommitOutput, FRIParams},
		reed_solomon::reed_solomon::ReedSolomonCode,
		transcript::ProverTranscript,
	};

	type U = OptimalUnderlier128b;
	type F = BinaryField128b;
	type FA = BinaryField16b;
	type P = PackedType<U, F>;

	/// Commits two random codewords f and f', runs the two-tree combination prover, and returns the
	/// finished transcript (the two roots and alpha are recoverable from it).
	fn prove_setup(
		seed: u64,
	) -> (
		FRIParams<F, FA>,
		BinaryMerkleTreeProver<F, Groestl256, Groestl256ByteCompression>,
		ProverTranscript<HasherChallenger<Groestl256>>,
	) {
		let log_dimension = 8;
		let log_inv_rate = 2;
		let log_batch_size = 0;
		let arities = [3usize, 2, 1];
		let n_test_queries = 5;

		let merkle_prover =
			BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);

		let rs_code = ReedSolomonCode::<FA>::new(log_dimension, log_inv_rate).unwrap();
		let params =
			FRIParams::new(rs_code, log_batch_size, arities.to_vec(), n_test_queries).unwrap();
		let rs_code = ReedSolomonCode::<FA>::new(log_dimension, log_inv_rate).unwrap();
		let ntt = SingleThreadedNTT::new(params.rs_code().log_len()).unwrap();

		let mut rng = StdRng::seed_from_u64(seed);
		let msg_len = rs_code.dim() << log_batch_size >> P::LOG_WIDTH;
		let msg_f: Vec<P> = repeat_with(|| P::random(&mut rng)).take(msg_len).collect();
		let msg_fprime: Vec<P> = repeat_with(|| P::random(&mut rng)).take(msg_len).collect();

		let CommitOutput { commitment: root_f, committed: committed_f, codeword: codeword_f } =
			fri::commit_interleaved(&rs_code, &params, &ntt, &merkle_prover, &msg_f).unwrap();
		let CommitOutput {
			commitment: root_fprime,
			committed: committed_fprime,
			codeword: codeword_fprime,
		} = fri::commit_interleaved(&rs_code, &params, &ntt, &merkle_prover, &msg_fprime).unwrap();

		let mut transcript = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		transcript.message().write(&root_f);
		transcript.message().write(&root_fprime);
		// alpha is sampled AFTER both f and f' are committed (bound into the transcript).
		let alpha: F = transcript.sample();

		prove_combination(
			&params,
			&ntt,
			&merkle_prover,
			&codeword_f,
			&committed_f,
			&codeword_fprime,
			&committed_fprime,
			alpha,
			&mut transcript,
		)
		.unwrap();

		(params, merkle_prover, transcript)
	}

	#[test]
	fn two_tree_combination_round_trips() {
		let (params, merkle_prover, transcript) = prove_setup(0);
		let mut verifier_transcript = transcript.into_verifier();

		let root_f = verifier_transcript.message().read().unwrap();
		let root_fprime = verifier_transcript.message().read().unwrap();
		let alpha: F = verifier_transcript.sample();

		verify_combination(
			&params,
			merkle_prover.scheme(),
			&root_f,
			&root_fprime,
			alpha,
			&mut verifier_transcript,
		)
		.expect("honest two-tree combination proof must verify");
	}

	#[test]
	fn wrong_alpha_is_rejected() {
		// The prover committed f, f' and combined = alpha*f + f'. A verifier using alpha' != alpha
		// must reject: the per-query cross-check combined_coset == alpha'*f + f' fails, so a combined
		// codeword is not bound to the pre-committed f, f' under the wrong challenge.
		let (params, merkle_prover, transcript) = prove_setup(0);
		let mut verifier_transcript = transcript.into_verifier();

		let root_f = verifier_transcript.message().read().unwrap();
		let root_fprime = verifier_transcript.message().read().unwrap();
		let alpha: F = verifier_transcript.sample();
		let wrong_alpha = alpha + <F as binius_field::Field>::ONE;

		let result = verify_combination(
			&params,
			merkle_prover.scheme(),
			&root_f,
			&root_fprime,
			wrong_alpha,
			&mut verifier_transcript,
		);
		assert!(result.is_err(), "a wrong combination challenge must be rejected");
	}
}
