// Copyright 2026 The pq-rollup Authors
// Addition to the Binius fork (Apache-2.0). See MODIFICATIONS.md.
//
// A2 (zero-knowledge roadmap) — porting Diamond's zero-knowledge Binius commitment
// (IACR ePrint 2025/1015, "Zero-Knowledge Polynomial Commitment in Binary Fields",
// Construction 4.1 "Zero-Knowledge Binary BaseFold"). That construction makes the large-field
// multilinear PCS hiding at the opened evaluations with two techniques from Aurora, carried through
// in characteristic 2:
//
//   1. High-end padding (this file). Set up the scheme over ℓ+1 variables and replace the input
//      f(X) with f(X) + Z_H(X)·r(X), where H is the fundamental domain and r is random. In the
//      Lin–Chung–Han novel polynomial basis this is arithmetic-free: it is exactly appending
//      `kappa` random high coefficients to the coefficient vector (Z_H(X) = W_ℓ(X) is a coordinate
//      up-shift). Since Z_H vanishes on H, evaluations on H — hence the multilinear evaluation the
//      PCS proves — are unchanged, while the disjoint Reed–Solomon domain that FRI queries is
//      randomised. `kappa = gamma · 2^vartheta` is the number of points each FRI oracle opens.
//
//   2. Second random polynomial + combination (this file). High padding alone is insufficient
//      because FRI folding shrinks the domain while the query count stays fixed; the remedy
//      (Aurora §5.1) is to sample a second *fully-random* polynomial f′ (same length as the padded
//      f) and run the FRI low-degree test on the virtual combination α·f + f′, where α is a verifier
//      challenge drawn from the large tower field L. Because RS encoding is F-linear, forming the
//      combination on coefficient vectors and encoding is identical to combining codewords, so we
//      combine on coefficients here. The FRI proximity on α·f + f′ certifies both f and f′ are
//      low-degree (Schwartz–Zippel over α: a non-low-degree f survives only with probability
//      1/|L| = 2^{-field_bits}), while the queried values reveal nothing about f because the
//      independent random f′ masks every opened point — this is the zero-knowledge of the query
//      phase. The `field_bits` term of `κ_ZK` (see the consumer repo's zk_accounting) is exactly
//      this α combination space.
//
// This file contributes techniques 1 and 2 as reusable, opt-in coefficient-domain helpers. The only
// remaining A2 step is wiring these through the BaseFold *evaluation* sumcheck so the
// transparency-to-evaluation is realised end to end (see paper/zk-a2-port-plan.md in the consumer
// repo). Neither helper changes an existing code path; the default (non-hiding) commitment is
// untouched.

use binius_field::Field;

/// Lay out the high-end-padded coefficient vector for a zero-knowledge commitment: the input
/// `message` (a power-of-two coefficient vector, the `ℓ`-variable witness) in the low half, `kappa`
/// fresh random coefficients in the next positions, and zeros filling the doubled `ℓ+1`-variable
/// space. `fill` writes the random coefficients (e.g. from an OS RNG); the helper carries no RNG
/// dependency.
///
/// The doubled length realises the `ℓ → ℓ+1` extension of Construction 4.1 (a Reed–Solomon code of
/// dimension `2^{ℓ+1}`), and the padding sits in the coordinates of `Z_H(X)·r(X)`, which vanish on
/// the fundamental domain — so the committed polynomial's evaluations on `H` (its multilinear
/// evaluation) are unchanged, while its values on the disjoint FRI query domain are randomised.
pub fn pad_message_high<F: Field>(message: &[F], kappa: usize, mut fill: impl FnMut(&mut [F])) -> Vec<F> {
	assert!(message.len().is_power_of_two(), "message length must be a power of two (2^ell)");
	assert!(kappa <= message.len(), "kappa random coefficients must fit the extra variable (kappa <= 2^ell)");

	let padded_len = message.len() * 2; // ell -> ell+1
	let mut out = vec![F::ZERO; padded_len];
	out[..message.len()].copy_from_slice(message);
	fill(&mut out[message.len()..message.len() + kappa]);
	out
}

/// Sample the second, fully-random masking polynomial `f′` for technique 2. It has the same length
/// as the (padded) committed polynomial `f`; `fill` writes every coefficient from a source of
/// randomness (the helper carries no RNG dependency). Unlike the high-end padding of technique 1,
/// `f′` is random in *all* coordinates — it is not constrained to vanish on the fundamental domain,
/// because it only participates in the low-degree test, never in the evaluation claim.
pub fn sample_mask_poly<F: Field>(len: usize, mut fill: impl FnMut(&mut [F])) -> Vec<F> {
	let mut out = vec![F::ZERO; len];
	fill(&mut out);
	out
}

/// Technique-2 combination: the virtual coefficient vector `α·f + f′` that the FRI low-degree test
/// is run on. `f` is the high-end-padded committed polynomial (technique 1), `f_prime` the
/// independent random masking polynomial ([`sample_mask_poly`]), and `alpha` a verifier challenge
/// from the large tower field. By F-linearity of Reed–Solomon encoding, combining on coefficients
/// and then encoding equals combining the codewords, so FRI proximity on this vector certifies both
/// `f` and `f_prime` are low-degree while the random `f_prime` hides `f` at the opened points.
pub fn combine_masked<F: Field>(f: &[F], f_prime: &[F], alpha: F) -> Vec<F> {
	assert_eq!(f.len(), f_prime.len(), "f and f' must share the padded length");
	f.iter().zip(f_prime).map(|(&fi, &gi)| alpha * fi + gi).collect()
}

/// Evaluation-domain view of technique-1 padding, for wiring into the BaseFold evaluation layer.
///
/// Where [`pad_message_high`] lays the padding out in the novel-polynomial (coefficient) basis that
/// FRI encodes, this lays it out over the Boolean hypercube that the BaseFold sumcheck evaluates: it
/// takes the `2^ell` hypercube evaluations of the committed multilinear `g` and returns the `2^{ell+1}`
/// evaluations of an extended multilinear `G` on `ell+1` variables, with the new variable most
/// significant — `G(x, 0) = g(x)` on the whole low half, and `kappa` random mask values in the high
/// half `G(x, 1)`. The two layouts coincide byte-for-byte ([data ‖ mask]) precisely because the novel
/// basis is triangular with respect to the subcube structure (this is why the padding is
/// arithmetic-free). Because `G` restricted to the new variable `= 0` is exactly `g`, every
/// evaluation the PCS actually proves — points of the form `(z, 0)` — is unchanged, while the FRI
/// query domain (the `= 1` slice) is randomised. `kappa <= 2^ell`.
pub fn extend_multilinear_zk<F: Field>(
	hypercube_evals: &[F],
	kappa: usize,
	mut fill: impl FnMut(&mut [F]),
) -> Vec<F> {
	assert!(hypercube_evals.len().is_power_of_two(), "evaluation count must be a power of two (2^ell)");
	assert!(kappa <= hypercube_evals.len(), "kappa mask values must fit the new variable (kappa <= 2^ell)");

	let n = hypercube_evals.len();
	let mut out = vec![F::ZERO; n * 2]; // ell -> ell+1, new variable most significant
	out[..n].copy_from_slice(hypercube_evals); // G(x, 0) = g(x)
	fill(&mut out[n..n + kappa]); // G(x, 1) = random mask on kappa points
	out
}

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField16b, Field};
	use rand::{rngs::StdRng, SeedableRng};

	use super::{combine_masked, extend_multilinear_zk, pad_message_high, sample_mask_poly};

	fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<BinaryField16b> {
		(0..n).map(|_| <BinaryField16b as Field>::random(&mut *rng)).collect()
	}

	#[test]
	fn high_end_padding_preserves_low_and_randomises_high() {
		let mut rng = StdRng::seed_from_u64(0xADD_1);
		let ell = 4usize;
		let n = 1 << ell; // 2^ell message coefficients
		let kappa = 5usize;

		let message: Vec<BinaryField16b> =
			(0..n).map(|_| <BinaryField16b as Field>::random(&mut rng)).collect();
		let padded = pad_message_high(&message, kappa, |buf| {
			for c in buf.iter_mut() {
				*c = <BinaryField16b as Field>::random(&mut rng);
			}
		});

		// Doubled length (ell -> ell+1), low half is the untouched message.
		assert_eq!(padded.len(), 2 * n);
		assert_eq!(&padded[..n], &message[..]);
		// The next `kappa` coefficients are the random padding; the remainder is zero.
		assert!(padded[n..n + kappa].iter().any(|&c| c != BinaryField16b::ZERO), "padding must be random");
		assert!(padded[n + kappa..].iter().all(|&c| c == BinaryField16b::ZERO), "beyond the padding must be zero");
	}

	#[test]
	fn combine_masked_matches_elementwise_and_is_linear() {
		let mut rng = StdRng::seed_from_u64(0xC0FFEE);
		let len = 32usize;
		let f = rand_vec(&mut rng, len);
		let f_prime = sample_mask_poly(len, |buf| {
			for c in buf.iter_mut() {
				*c = <BinaryField16b as Field>::random(&mut rng);
			}
		});
		let alpha = <BinaryField16b as Field>::random(&mut rng);

		let combined = combine_masked(&f, &f_prime, alpha);
		assert_eq!(combined.len(), len);

		// Elementwise identity: combined[i] == alpha*f[i] + f'[i].
		for i in 0..len {
			assert_eq!(combined[i], alpha * f[i] + f_prime[i]);
			// Mask is recoverable without a field inverse: combined - f' == alpha*f
			// (subtraction is addition in characteristic 2). This is the simulator's handle.
			assert_eq!(combined[i] + f_prime[i], alpha * f[i]);
		}

		// The combination commutes with any linear functional (here, the coefficient sum), which is
		// why FRI proximity on alpha*f + f' certifies both f and f' at once.
		let sum = |v: &[BinaryField16b]| v.iter().copied().fold(BinaryField16b::ZERO, |a, b| a + b);
		assert_eq!(sum(&combined), alpha * sum(&f) + sum(&f_prime));
	}

	#[test]
	fn extend_multilinear_preserves_evaluation_on_the_hypercube() {
		use binius_field::BinaryField128b;
		use binius_math::{MultilinearExtension, MultilinearQuery};

		type F = BinaryField128b;
		let mut rng = StdRng::seed_from_u64(0xBA5E_F01D);
		let ell = 6usize;
		let n = 1usize << ell;
		let kappa = 12usize; // e.g. the FRI query count

		// Original committed multilinear g on `ell` variables (hypercube evaluations).
		let g_evals: Vec<F> = (0..n).map(|_| <F as Field>::random(&mut rng)).collect();

		// Zero-knowledge extension G on `ell+1` variables: low half = g, high half = kappa masks.
		let g_ext = extend_multilinear_zk(&g_evals, kappa, |buf| {
			for c in buf.iter_mut() {
				*c = <F as Field>::random(&mut rng);
			}
		});
		assert_eq!(g_ext.len(), 2 * n);

		let g = MultilinearExtension::from_values(g_evals.clone()).unwrap();
		let g_big = MultilinearExtension::from_values(g_ext.clone()).unwrap();
		let mask = MultilinearExtension::from_values(g_ext[n..].to_vec()).unwrap();

		// A random evaluation point z in the original ell variables.
		let z: Vec<F> = (0..ell).map(|_| <F as Field>::random(&mut rng)).collect();

		let eval_g = g.evaluate(MultilinearQuery::<F>::expand(&z).to_ref()).unwrap();

		// THE invariant: evaluating the extension at (z, new_var = 0) reproduces g(z) exactly —
		// the evaluation the PCS proves is untouched by the zero-knowledge padding.
		let mut z0 = z.clone();
		z0.push(F::ZERO);
		let eval_ext_0: F = g_big.evaluate(MultilinearQuery::<F>::expand(&z0).to_ref()).unwrap();
		assert_eq!(eval_ext_0, eval_g, "padding must preserve the hypercube evaluation");

		// And at (z, new_var = 1) the extension is exactly the mask multilinear — the live
		// randomness that hides the FRI query domain. (Exact identity, not probabilistic.)
		let mut z1 = z.clone();
		z1.push(F::ONE);
		let eval_ext_1: F = g_big.evaluate(MultilinearQuery::<F>::expand(&z1).to_ref()).unwrap();
		let eval_mask = mask.evaluate(MultilinearQuery::<F>::expand(&z).to_ref()).unwrap();
		assert_eq!(eval_ext_1, eval_mask, "the new-variable=1 slice is exactly the mask");
		assert_ne!(eval_ext_1, eval_g, "the mask slice is live (differs from g)");
	}

	#[test]
	fn fri_fold_is_linear_so_the_alpha_combination_opens_from_two_trees() {
		// Technique 2 runs the FRI low-degree test on the virtual combination alpha*f + f', with f
		// and f' committed in separate Merkle trees and alpha sampled after both commitments. For the
		// verifier to open f and f' separately and reconstruct (alpha*f + f') at the query positions,
		// the FRI fold map must be F-linear in the codeword. This checks that invariant against the
		// REAL binius `fold_codeword`, so the two-tree opening is sound by construction.
		use binius_field::{BinaryField128b, BinaryField16b, Field};
		use binius_ntt::SingleThreadedNTT;

		use crate::{protocols::fri::fold_codeword, reed_solomon::reed_solomon::ReedSolomonCode};

		type F = BinaryField128b; // large tower field the codeword lives in
		type FS = BinaryField16b; // Reed–Solomon encoding subfield

		let mut rng = StdRng::seed_from_u64(0xF01D_11);
		let log_dim = 8usize;
		let log_inv_rate = 1usize;
		let rs_code = ReedSolomonCode::<FS>::new(log_dim, log_inv_rate).unwrap();
		let ntt = SingleThreadedNTT::<FS>::new(rs_code.log_len()).unwrap();

		let len = 1usize << rs_code.log_len(); // codeword length
		let c_f: Vec<F> = (0..len).map(|_| <F as Field>::random(&mut rng)).collect();
		let c_fprime: Vec<F> = (0..len).map(|_| <F as Field>::random(&mut rng)).collect();
		let alpha = <F as Field>::random(&mut rng);

		// The virtual combined codeword the low-degree test is actually run on.
		let c_combined = combine_masked(&c_f, &c_fprime, alpha);

		// Fold all three with the same challenges via the real FRI fold primitive.
		let n_fold = 3usize;
		let challenges: Vec<F> = (0..n_fold).map(|_| <F as Field>::random(&mut rng)).collect();
		let fold = |cw: &[F]| fold_codeword(&ntt, &rs_code, cw, n_fold, &challenges);

		let folded_combined = fold(&c_combined);
		let folded_f = fold(&c_f);
		let folded_fprime = fold(&c_fprime);

		// fold(alpha*f + f') == alpha*fold(f) + fold(f'): folding a combination equals combining the
		// folds, so a verifier holding the per-tree openings can reconstruct the combined fold.
		let recombined = combine_masked(&folded_f, &folded_fprime, alpha);
		assert_eq!(folded_combined, recombined, "FRI fold must be linear in the codeword");
		assert_eq!(folded_combined.len(), len >> n_fold);
	}

	#[test]
	fn combine_masked_degenerate_challenges() {
		let mut rng = StdRng::seed_from_u64(0xD00D);
		let len = 16usize;
		let f = rand_vec(&mut rng, len);
		let f_prime = rand_vec(&mut rng, len);

		// alpha = 0 -> pure mask f' (reveals nothing about f).
		assert_eq!(combine_masked(&f, &f_prime, BinaryField16b::ZERO), f_prime);
		// alpha = 1, f' = 0 -> exactly f.
		let zeros = vec![BinaryField16b::ZERO; len];
		assert_eq!(combine_masked(&f, &zeros, BinaryField16b::ONE), f);
	}
}
