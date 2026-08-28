// Copyright 2025 Anonymous.

//! A Binius SNARK gadget that proves execution of Poseidon2b permutations -- the *binary-field* version
//! of Poseidon2 (Grassi--Khovratovich--Koschatko--Rechberger--Schofnegger--Schroeppel--Wu, IACR ePrint
//! 2025/1893), designed specifically for binary-tower proving systems such as Binius.
//!
//! This gadget exists to close the last open cost comparison in the recursion paper: FIPS Keccak-f vs.
//! the arithmetization-friendly (AO) hashes over binary towers. The A/B (`ao_ab_bench`) already pits
//! Keccak-f against Vision Mark-32 and Blake3; this adds the Poseidon2b arm through the *identical*
//! proving path (same `ConstraintSystemBuilder`, canonical B128 tower, Groestl256 FRI/Merkle).
//!
//! ## Structural fidelity vs. cryptographic instantiation
//! The arithmetization here is *structurally* faithful to the headline `x7_32_512` instance:
//!   * field GF(2^32) (B32 tower level), state width `t = 16` (512-bit state);
//!   * S-box x^7, arithmetized as the degree-2 chain a -> a^2 -> a^4 -> a^6 -> a^7 (2 committed squares
//!     + 2 committed products per S-box), exactly as in the reference `poseidon2b_x7_32_512.rs`;
//!   * HADES round schedule R_F = 10 (5 + 5) full rounds and R_P = 15 partial rounds (25 rounds total);
//!   * external (full-round) 16x16 MDS layer + internal (partial-round) matrix M_I = diag(mu) + all-ones
//!     (`y_i = mu_i * x_i + sum_{j != i} x_j`, the O(t) Poseidon2 internal layer).
//!
//! The round-constant values and MDS/mu entries below are self-consistent PLACEHOLDERS: prover and
//! verifier cost depend only on the arithmetization *shape* (column count, constraint count/degree),
//! not on the specific field constants, so the measured ms/perm is the true cost of a Poseidon2b
//! permutation of this shape. It is NOT a cryptographic instantiation (these are not the secure
//! constants), and no security claim is attached to the constants -- only to the cost measurement.

use std::array;

use anyhow::Result;
use binius_core::oracle::OracleId;
use binius_field::{BinaryField32b, Field, TowerField};
use binius_math::ArithExpr;

use crate::builder::{types::F, ConstraintSystemBuilder};

type B32 = BinaryField32b;

pub const STATE_SIZE: usize = 16;
const R_F_HALF: usize = 5; // half of R_F = 10 full rounds
const R_P: usize = 15; // partial rounds

/// Internal-matrix diagonal (mu_i): 16 distinct nonzero B32 constants (shape-faithful placeholder,
/// mirroring the small-value diagonal `0x400, 0x8, 0x100, ...` of the reference internal matrix).
#[rustfmt::skip]
const MU: [u32; STATE_SIZE] = [
    0x400, 0x8, 0x100, 0x800, 0x2, 0x40, 0x1000, 0x20,
    0x4, 0x200, 0x2000, 0x10, 0x80, 0x4000, 0x1, 0x8000,
];

/// First row of the external (full-round) 16x16 MDS layer; the full matrix is its circulant, so every
/// entry is nonzero (a dense external MDS -- the expensive, faithful case).
#[rustfmt::skip]
const ME_ROW: [u32; STATE_SIZE] = [
    0x0a, 0x0e, 0x02, 0x06, 0x03, 0x07, 0x0b, 0x0f,
    0x05, 0x0d, 0x09, 0x01, 0x0c, 0x04, 0x08, 0x11,
];

/// External MDS entry M_E[i][j] via the circulant of `ME_ROW`.
#[inline]
fn me(i: usize, j: usize) -> B32 {
    B32::new(ME_ROW[(j + STATE_SIZE - i) % STATE_SIZE])
}

/// Round constant RC[round][lane] -- deterministic self-consistent placeholder (cost-invariant).
#[inline]
fn rc(round: usize, lane: usize) -> u32 {
    let mut x = (round as u32)
        .wrapping_mul(0x9E37_79B1)
        ^ (lane as u32).wrapping_mul(0x85EB_CA77)
        ^ 0xABCD_1234;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2C1B_3C6D);
    x ^= x >> 12;
    x | 1 // keep nonzero
}

// t = a^2 : oracle_ids [a, t], expr = Var(1) - Var(0)^2
fn square_expr() -> ArithExpr<B32> {
    ArithExpr::Var(1) - ArithExpr::Var(0).pow(2)
}

// p = x*y : oracle_ids [x, y, p], expr = Var(2) - Var(0)*Var(1)
fn mul_expr() -> ArithExpr<B32> {
    ArithExpr::Var(2) - ArithExpr::Var(0) * ArithExpr::Var(1)
}

/// Constrain a single x^7 S-box on input oracle `a_id` (with input value `a_val` supplied at witness
/// time). Commits the degree-2 chain a^2, a^4, a^6, a^7 and returns the output oracle id (a^7).
#[allow(clippy::too_many_arguments)]
fn sbox(
    builder: &mut ConstraintSystemBuilder,
    log_size: usize,
    tag: &str,
    a_id: OracleId,
) -> Result<OracleId> {
    let t2 = builder.add_committed(format!("{tag}.a2"), log_size, B32::TOWER_LEVEL);
    let t4 = builder.add_committed(format!("{tag}.a4"), log_size, B32::TOWER_LEVEL);
    let t6 = builder.add_committed(format!("{tag}.a6"), log_size, B32::TOWER_LEVEL);
    let o = builder.add_committed(format!("{tag}.a7"), log_size, B32::TOWER_LEVEL);

    if let Some(witness) = builder.witness() {
        let a_val = witness.get::<B32>(a_id)?;
        let a_val = a_val.as_slice::<B32>();
        let mut t2_c = witness.new_column::<B32>(t2);
        let mut t4_c = witness.new_column::<B32>(t4);
        let mut t6_c = witness.new_column::<B32>(t6);
        let mut o_c = witness.new_column::<B32>(o);
        let t2_s = t2_c.as_mut_slice::<B32>();
        let t4_s = t4_c.as_mut_slice::<B32>();
        let t6_s = t6_c.as_mut_slice::<B32>();
        let o_s = o_c.as_mut_slice::<B32>();
        for z in 0..1 << log_size {
            let a = a_val[z];
            let a2 = a * a;
            let a4 = a2 * a2;
            let a6 = a2 * a4;
            let a7 = a6 * a;
            t2_s[z] = a2;
            t4_s[z] = a4;
            t6_s[z] = a6;
            o_s[z] = a7;
        }
    }

    builder.assert_zero(format!("{tag}.sq2"), [a_id, t2], square_expr().convert_field());
    builder.assert_zero(format!("{tag}.sq4"), [t2, t4], square_expr().convert_field());
    builder.assert_zero(format!("{tag}.mul6"), [t2, t4, t6], mul_expr().convert_field());
    builder.assert_zero(format!("{tag}.mul7"), [t6, a_id, o], mul_expr().convert_field());
    Ok(o)
}

/// Build a linear-combination oracle (virtual, no zerocheck) from `terms` plus a constant `offset`,
/// and populate its witness column. `vals` supplies, per input oracle, its already-known value slice
/// index-aligned with `terms`; the offset is a B32 constant.
fn lin(
    builder: &mut ConstraintSystemBuilder,
    log_size: usize,
    name: String,
    terms: &[(OracleId, B32)],
    offset: B32,
) -> Result<OracleId> {
    let id = builder.add_linear_combination_with_offset(
        name,
        log_size,
        F::from(offset),
        terms.iter().map(|&(o, c)| (o, F::from(c))),
    )?;
    if let Some(witness) = builder.witness() {
        // Read each input slice, then write the combination.
        let inputs: Vec<_> = terms
            .iter()
            .map(|&(o, c)| Ok::<_, anyhow::Error>((witness.get::<B32>(o)?, c)))
            .collect::<Result<_, _>>()?;
        let input_slices: Vec<(&[B32], B32)> =
            inputs.iter().map(|(w, c)| (w.as_slice::<B32>(), *c)).collect();
        let mut col = witness.new_column::<B32>(id);
        let out = col.as_mut_slice::<B32>();
        for z in 0..1 << log_size {
            let mut acc = offset;
            for (s, c) in &input_slices {
                acc += *c * s[z];
            }
            out[z] = acc;
        }
    }
    Ok(id)
}

/// Prove `2^log_size` Poseidon2b permutations of the all-`p_in` input state. Returns the output state
/// oracle ids. Mirrors the `vision_permutation` gadget shape so it drops into the same A/B harness.
pub fn poseidon2b_permutation(
    builder: &mut ConstraintSystemBuilder,
    log_size: usize,
    p_in: [OracleId; STATE_SIZE],
) -> Result<[OracleId; STATE_SIZE]> {
    // --- initial external MDS layer: state = M_E * p_in ---
    let mut state: [OracleId; STATE_SIZE] = array::from_fn(|i| {
        let terms: Vec<_> = (0..STATE_SIZE).map(|j| (p_in[j], me(i, j))).collect();
        lin(builder, log_size, format!("init_mds[{i}]"), &terms, B32::ZERO).expect("init mds")
    });

    let mut round: usize = 0;

    // --- first R_F/2 full rounds ---
    for _ in 0..R_F_HALF {
        state = full_round(builder, log_size, round, state)?;
        round += 1;
    }
    // --- R_P partial rounds ---
    for _ in 0..R_P {
        state = partial_round(builder, log_size, round, state)?;
        round += 1;
    }
    // --- last R_F/2 full rounds ---
    for _ in 0..R_F_HALF {
        state = full_round(builder, log_size, round, state)?;
        round += 1;
    }

    Ok(state)
}

fn full_round(
    builder: &mut ConstraintSystemBuilder,
    log_size: usize,
    round: usize,
    state: [OracleId; STATE_SIZE],
) -> Result<[OracleId; STATE_SIZE]> {
    builder.push_namespace(format!("full[{round}]"));
    // add round constant (into S-box input) + S-box each lane
    let mut o: [OracleId; STATE_SIZE] = array::from_fn(|_| usize::MAX);
    for i in 0..STATE_SIZE {
        let a = lin(
            builder,
            log_size,
            format!("a[{i}]"),
            &[(state[i], B32::ONE)],
            B32::new(rc(round, i)),
        )?;
        o[i] = sbox(builder, log_size, &format!("sbox[{i}]"), a)?;
    }
    // external MDS: next[i] = sum_j M_E[i][j] * o[j]
    let next: [OracleId; STATE_SIZE] = array::from_fn(|i| {
        let terms: Vec<_> = (0..STATE_SIZE).map(|j| (o[j], me(i, j))).collect();
        lin(builder, log_size, format!("mds[{i}]"), &terms, B32::ZERO).expect("mds")
    });
    builder.pop_namespace();
    Ok(next)
}

fn partial_round(
    builder: &mut ConstraintSystemBuilder,
    log_size: usize,
    round: usize,
    state: [OracleId; STATE_SIZE],
) -> Result<[OracleId; STATE_SIZE]> {
    builder.push_namespace(format!("partial[{round}]"));
    // S-box on lane 0 only
    let a0 = lin(
        builder,
        log_size,
        "a0".to_string(),
        &[(state[0], B32::ONE)],
        B32::new(rc(round, 0)),
    )?;
    let o0 = sbox(builder, log_size, "sbox0", a0)?;
    // x = (o0, state[1], .., state[15]); internal MDS: next[i] = mu_i * x[i] + sum_{j!=i} x[j]
    let x: [OracleId; STATE_SIZE] =
        array::from_fn(|j| if j == 0 { o0 } else { state[j] });
    let next: [OracleId; STATE_SIZE] = array::from_fn(|i| {
        let terms: Vec<_> = (0..STATE_SIZE)
            .map(|j| (x[j], if j == i { B32::new(MU[i]) } else { B32::ONE }))
            .collect();
        lin(builder, log_size, format!("imds[{i}]"), &terms, B32::ZERO).expect("imds")
    });
    builder.pop_namespace();
    Ok(next)
}

#[cfg(test)]
mod tests {
    use binius_field::BinaryField32b;

    use super::{poseidon2b_permutation, STATE_SIZE};
    use crate::{builder::test_utils::test_circuit, unconstrained::unconstrained};

    #[test]
    fn test_poseidon2b() {
        test_circuit(|builder| {
            let log_size = 8;
            let state_in: [_; STATE_SIZE] = std::array::from_fn(|i| {
                unconstrained::<BinaryField32b>(builder, format!("p_in[{i}]"), log_size).unwrap()
            });
            let _state_out = poseidon2b_permutation(builder, log_size, state_in).unwrap();
            Ok(vec![])
        })
        .unwrap();
    }
}
