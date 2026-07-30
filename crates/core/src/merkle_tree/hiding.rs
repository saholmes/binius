// Copyright 2026 The pq-rollup Authors
// Addition to the Binius fork (Apache-2.0). See MODIFICATIONS.md.
//
// A1 (zero-knowledge roadmap): a **hiding (salted) binary Merkle commitment**.
//
// A leaf digest is `H(real_elems ‖ salt_elems)`. By expressing the per-leaf salt as extra field
// elements appended to each leaf's committed batch, the commitment is hiding *without any change to
// the Merkle scheme*: the existing `build` / `verify_opening` hash whatever elements are committed,
// and the salt is revealed as part of the opened values on opening. So this is an entirely opt-in
// layer over the unmodified [`super::BinaryMerkleTreeProver`] — the default (non-hiding) commitment
// path is untouched.
//
// Hiding rests on the salt's entropy: for a fresh, uniformly random salt (drawn by the caller, e.g.
// from the OS or a FIPS-202 XOF), two commitments to the *same* real data under independent salts
// have independent, unrelated roots. (Hiding the *opened* leaf values against a query-bounded
// verifier is a separate concern handled by trace masking / a masked low-degree test — see the ZK
// roadmap; this module provides the commitment-hiding building block.)

use binius_field::TowerField;

/// Interleave a per-leaf `salt` into the per-leaf `real` data, producing the salted leaf batches a
/// hiding commitment commits to.
///
/// Both inputs are leaf-major: `real` is `n_leaves × real_len` and `salt` is `n_leaves × salt_len`,
/// and the output is `n_leaves × (real_len + salt_len)` with each leaf's real elements followed by
/// its salt. Commit the result with [`super::BinaryMerkleTreeProver::commit`] at batch size
/// `real_len + salt_len`; open and verify exactly as usual, passing each opened leaf's full
/// `real ‖ salt` values.
pub fn salt_leaves<F: TowerField>(real: &[F], real_len: usize, salt: &[F], salt_len: usize) -> Vec<F> {
	assert!(real_len > 0, "real batch must be non-empty");
	assert!(real.len() % real_len == 0, "real data must be a whole number of leaves");
	let n_leaves = real.len() / real_len;
	assert!(salt.len() == n_leaves * salt_len, "salt must supply salt_len elements per leaf");

	let mut out = Vec::with_capacity(real.len() + salt.len());
	for i in 0..n_leaves {
		out.extend_from_slice(&real[i * real_len..(i + 1) * real_len]);
		out.extend_from_slice(&salt[i * salt_len..(i + 1) * salt_len]);
	}
	out
}

/// The full committed values (`real ‖ salt`) of one leaf — what an opening reveals and the verifier
/// hashes. Convenience for assembling the `values` argument to `verify_opening`.
pub fn leaf_values<F: TowerField>(real: &[F], real_len: usize, salt: &[F], salt_len: usize, index: usize) -> Vec<F> {
	let mut vals = Vec::with_capacity(real_len + salt_len);
	vals.extend_from_slice(&real[index * real_len..(index + 1) * real_len]);
	vals.extend_from_slice(&salt[index * salt_len..(index + 1) * salt_len]);
	vals
}

#[cfg(test)]
mod tests {
	use std::{iter::repeat_with, slice};

	use binius_field::{BinaryField16b, Field};
	use binius_hash::groestl::{Groestl256, Groestl256ByteCompression};
	use rand::{rngs::StdRng, SeedableRng};

	use super::{leaf_values, salt_leaves};
	use crate::{
		fiat_shamir::HasherChallenger,
		merkle_tree::{BinaryMerkleTreeProver, MerkleTreeProver, MerkleTreeScheme},
		transcript::ProverTranscript,
	};

	#[test]
	fn hiding_salted_commitment_is_hiding_and_opens() {
		let mut rng = StdRng::seed_from_u64(0);
		let prover = BinaryMerkleTreeProver::<_, Groestl256, _>::new(Groestl256ByteCompression);

		let n_leaves = 16usize;
		let (real_len, salt_len) = (1usize, 2usize);
		let tree_depth = 4; // log2(16)

		let real: Vec<BinaryField16b> = repeat_with(|| Field::random(&mut rng)).take(n_leaves * real_len).collect();
		let salt_a: Vec<BinaryField16b> = repeat_with(|| Field::random(&mut rng)).take(n_leaves * salt_len).collect();
		let salt_b: Vec<BinaryField16b> = repeat_with(|| Field::random(&mut rng)).take(n_leaves * salt_len).collect();

		let batch = real_len + salt_len;
		let (commit_a, tree_a) = prover.commit(&salt_leaves(&real, real_len, &salt_a, salt_len), batch).unwrap();
		let (commit_b, _) = prover.commit(&salt_leaves(&real, real_len, &salt_b, salt_len), batch).unwrap();

		// Hiding: the SAME real data under independent salts gives different commitment roots.
		assert_ne!(commit_a.root, commit_b.root, "commitment must hide the data (salt randomises the root)");
		// Determinism: the same real data + same salt gives the same root.
		let (commit_a2, _) = prover.commit(&salt_leaves(&real, real_len, &salt_a, salt_len), batch).unwrap();
		assert_eq!(commit_a.root, commit_a2.root);

		// Open + verify every leaf, revealing (real ‖ salt) as the leaf values.
		for i in 0..n_leaves {
			let mut writer = ProverTranscript::<HasherChallenger<Groestl256>>::new();
			prover.prove_opening(&tree_a, 0, i, &mut writer.message()).unwrap();
			let mut reader = writer.into_verifier();
			let vals = leaf_values(&real, real_len, &salt_a, salt_len, i);
			prover
				.scheme()
				.verify_opening(i, &vals, 0, tree_depth, &[commit_a.root], &mut reader.message())
				.unwrap();
		}

		// Opening with the WRONG salt (or wrong real value) is rejected.
		let mut writer = ProverTranscript::<HasherChallenger<Groestl256>>::new();
		prover.prove_opening(&tree_a, 0, 0, &mut writer.message()).unwrap();
		let mut reader = writer.into_verifier();
		let mut wrong = slice::from_ref(&real[0]).to_vec();
		wrong.extend_from_slice(&salt_b[0..salt_len]); // salt from the other commitment
		assert!(
			prover
				.scheme()
				.verify_opening(0, &wrong, 0, tree_depth, &[commit_a.root], &mut reader.message())
				.is_err(),
			"a leaf opened with the wrong salt must fail verification"
		);
	}
}
