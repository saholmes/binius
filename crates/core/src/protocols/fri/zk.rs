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
//   2. Second random polynomial + combination (NOT in this file; next step). High padding alone is
//      insufficient because FRI folding shrinks the domain while the query count stays fixed; the
//      remedy (Aurora §5.1) is to sample a second fully-random polynomial f′ and run FRI on the
//      virtual combination α·f + f′ (α a verifier challenge). This, plus wiring the padding through
//      the BaseFold *evaluation* sumcheck so the transparency-to-evaluation is realised end to end,
//      is the remaining A2 work (see paper/zk-a2-port-plan.md in the consumer repo).
//
// This file contributes technique 1's coefficient-vector layout as a reusable, opt-in helper. It
// changes no existing code path; the default (non-hiding) commitment is untouched.

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

#[cfg(test)]
mod tests {
	use binius_field::{BinaryField16b, Field};
	use rand::{rngs::StdRng, SeedableRng};

	use super::pad_message_high;

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
}
