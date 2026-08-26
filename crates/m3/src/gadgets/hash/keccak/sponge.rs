// Copyright 2025 Irreducible Inc.
// Added 2025 by Anonymous (Apache-2.0 derivative): a multi-block Keccak sponge gadget with a
// CONSTRAINED absorb chain — the "bindable sponge". It links successive Keccak-f permutations
// through the packed state using the same masked-equality technique the `Keccakf` gadget uses for
// its own 8-track round pipeline (`link_out_to_next_in`), so the sponge absorb is a first-class
// constraint and no `Selected`/Projected extraction from the packed state is required.

use anyhow::Result;
use binius_core::oracle::ShiftVariant;
use binius_field::{
	as_packed_field::PackScalar, underlier::WithUnderlier, ExtensionField, Field, PackedExtension,
	PackedFieldIndexable, TowerField,
};
use bytemuck::Pod;

use super::{Keccakf, PackedLane8, StateMatrix};
use crate::builder::{TableBuilder, TableWitnessSegment, B1, B8};

/// Bits per lane, and the number of tracks packed into a [`PackedLane8`] (the permutation-input
/// track is 0, the permutation-output track is 7).
const LANE_BITS: usize = 64;
const TRACKS: usize = 8;
/// Number of lanes in the full Keccak state.
const STATE_LANES: usize = 25;
/// Shift (in bits) that aligns the output track (7) to the input track (0): `7 * 64`.
const OUT_TO_IN_SHIFT: usize = LANE_BITS * (TRACKS - 1);
/// `log2` of the packed-lane width in bits (`512`), i.e. the shift block size covering the lane.
const PACKED_LOG_BITS: usize = 9;

/// A multi-block Keccak sponge with a **constrained** absorb chain.
///
/// Wraps `n_blocks` [`Keccakf`] permutations. Permutation `b`'s input state (track 0 of its
/// `state_in`) is constrained to equal permutation `b - 1`'s output state (track 7 of its
/// `state_out`, shifted into track 0) XOR message block `b` (over the rate lanes); block `0`
/// absorbs into the zero state. The per-block message columns are exposed via [`Self::message`] so
/// the caller can bind them to the message source; the sponge output (final permutation, track 7)
/// is exposed via [`Self::output`].
///
/// The constraints only touch track 0 (the permutation input); the permutations' internal round
/// pipelines (tracks 1..8) are left to the [`Keccakf`] gadget. The absorb equality is masked to
/// track 0 with a constant selector — the same construction as [`Keccakf`]'s `link_out_to_next_in`.
pub struct KeccakSponge {
	perms: Vec<Keccakf>,
	message: Vec<StateMatrix<PackedLane8>>,
	/// Previous-output-aligned-to-input columns, one per block `b >= 1` (indexed by `b - 1`).
	prev_shifts: Vec<StateMatrix<PackedLane8>>,
	track0_mask: PackedLane8,
	rate_lanes: usize,
	/// For a SHARDED sponge (a block-run of a larger sponge): the injected initial state (track 0),
	/// bound by the caller to a public value. `None` for a standard sponge (absorbs into the zero state).
	initial_state: Option<StateMatrix<PackedLane8>>,
}

impl KeccakSponge {
	/// Build a sponge of `n_blocks` blocks with sponge rate `rate_lanes` (17 lanes = 136 bytes for
	/// SHA3-256 / SHAKE-256). Adds the permutations, the per-block message columns, and the absorb
	/// chain constraints to `table`.
	pub fn new<F>(table: &mut TableBuilder<F>, n_blocks: usize, rate_lanes: usize) -> Self
	where
		F: TowerField + ExtensionField<B1>,
		<F as WithUnderlier>::Underlier: PackScalar<B1>,
	{
		assert!(n_blocks >= 1, "a sponge needs at least one block");
		assert!(rate_lanes <= STATE_LANES, "rate cannot exceed the state");

		// Selector with all ones in the input track (bits [0, 64)) and zero elsewhere.
		let track0_mask: PackedLane8 = table.add_constant(
			"sponge_track0_mask",
			std::array::from_fn(|bit| if bit < LANE_BITS { B1::ONE } else { B1::ZERO }),
		);

		let mut perms: Vec<Keccakf> = Vec::with_capacity(n_blocks);
		let mut message: Vec<StateMatrix<PackedLane8>> = Vec::with_capacity(n_blocks);
		let mut prev_shifts: Vec<StateMatrix<PackedLane8>> = Vec::with_capacity(n_blocks.saturating_sub(1));

		for b in 0..n_blocks {
			let state_in = StateMatrix::from_fn(|(x, y)| {
				table.add_committed::<B1, { LANE_BITS * TRACKS }>(format!("sponge_state_in[{b}][{x},{y}]"))
			});
			let kf = Keccakf::new(&mut table.with_namespace(format!("sponge_perm[{b}]")), state_in);
			let msg = StateMatrix::from_fn(|(x, y)| {
				table.add_committed::<B1, { LANE_BITS * TRACKS }>(format!("sponge_message[{b}][{x},{y}]"))
			});

			let state_in = kf.packed_state_in().clone();
			if b == 0 {
				// Absorb into the zero state: input track 0 = message block.
				for x in 0..5 {
					for y in 0..5 {
						table.assert_zero(
							"sponge_absorb_first",
							(state_in[(x, y)] - msg[(x, y)]) * track0_mask,
						);
					}
				}
			} else {
				// Align the previous permutation output (track 7) to the input track (track 0).
				let prev_out = perms[b - 1].packed_state_out().clone();
				let prev_shift = StateMatrix::from_fn(|(x, y)| {
					table.add_shifted(
						format!("sponge_prev_shift[{b}][{x},{y}]"),
						prev_out[(x, y)],
						PACKED_LOG_BITS,
						OUT_TO_IN_SHIFT,
						ShiftVariant::LogicalRight,
					)
				});
				// Absorb: input track 0 = previous output XOR message block (XOR = + over GF(2)).
				for x in 0..5 {
					for y in 0..5 {
						table.assert_zero(
							"sponge_absorb",
							(state_in[(x, y)] - prev_shift[(x, y)] - msg[(x, y)]) * track0_mask,
						);
					}
				}
				prev_shifts.push(prev_shift);
			}

			perms.push(kf);
			message.push(msg);
		}

		Self { perms, message, prev_shifts, track0_mask, rate_lanes, initial_state: None }
	}

	/// Build a **sharded** sponge: a block-run of a larger sponge that absorbs its first block into an
	/// INJECTED initial state instead of the zero state. The injected state (track 0 of each lane) is a fresh
	/// committed [`StateMatrix`] the caller must bind to a public value (the previous shard's output state);
	/// obtain it via [`Self::initial_state`]. The last block's output ([`Self::output`]) is this shard's
	/// output state, to be pinned public and fed to the next shard. Block-wise sharding lets one huge sponge
	/// be proved in Pi-sized pieces, chained by their 25-lane Keccak states across proofs.
	pub fn new_with_initial_state<F>(table: &mut TableBuilder<F>, n_blocks: usize, rate_lanes: usize) -> Self
	where
		F: TowerField + ExtensionField<B1>,
		<F as WithUnderlier>::Underlier: PackScalar<B1>,
	{
		assert!(n_blocks >= 1, "a sponge needs at least one block");
		assert!(rate_lanes <= STATE_LANES, "rate cannot exceed the state");

		let track0_mask: PackedLane8 = table.add_constant(
			"sponge_track0_mask",
			std::array::from_fn(|bit| if bit < LANE_BITS { B1::ONE } else { B1::ZERO }),
		);
		// The injected initial state (track 0): a committed state the caller binds to the previous shard's out.
		let init: StateMatrix<PackedLane8> = StateMatrix::from_fn(|(x, y)| {
			table.add_committed::<B1, { LANE_BITS * TRACKS }>(format!("sponge_initial_state[{x},{y}]"))
		});

		let mut perms: Vec<Keccakf> = Vec::with_capacity(n_blocks);
		let mut message: Vec<StateMatrix<PackedLane8>> = Vec::with_capacity(n_blocks);
		let mut prev_shifts: Vec<StateMatrix<PackedLane8>> = Vec::with_capacity(n_blocks.saturating_sub(1));

		for b in 0..n_blocks {
			let state_in = StateMatrix::from_fn(|(x, y)| {
				table.add_committed::<B1, { LANE_BITS * TRACKS }>(format!("sponge_state_in[{b}][{x},{y}]"))
			});
			let kf = Keccakf::new(&mut table.with_namespace(format!("sponge_perm[{b}]")), state_in);
			let msg = StateMatrix::from_fn(|(x, y)| {
				table.add_committed::<B1, { LANE_BITS * TRACKS }>(format!("sponge_message[{b}][{x},{y}]"))
			});
			let state_in = kf.packed_state_in().clone();
			if b == 0 {
				// Absorb first block into the INJECTED state: input track 0 = initial_state XOR message.
				for x in 0..5 {
					for y in 0..5 {
						table.assert_zero(
							"sponge_absorb_first_injected",
							(state_in[(x, y)] - init[(x, y)] - msg[(x, y)]) * track0_mask,
						);
					}
				}
			} else {
				let prev_out = perms[b - 1].packed_state_out().clone();
				let prev_shift = StateMatrix::from_fn(|(x, y)| {
					table.add_shifted(
						format!("sponge_prev_shift[{b}][{x},{y}]"),
						prev_out[(x, y)],
						PACKED_LOG_BITS,
						OUT_TO_IN_SHIFT,
						ShiftVariant::LogicalRight,
					)
				});
				for x in 0..5 {
					for y in 0..5 {
						table.assert_zero(
							"sponge_absorb",
							(state_in[(x, y)] - prev_shift[(x, y)] - msg[(x, y)]) * track0_mask,
						);
					}
				}
				prev_shifts.push(prev_shift);
			}
			perms.push(kf);
			message.push(msg);
		}
		Self { perms, message, prev_shifts, track0_mask, rate_lanes, initial_state: Some(init) }
	}

	/// The injected initial-state columns (track 0), present only for a sharded sponge built via
	/// [`Self::new_with_initial_state`]. Bind these to the previous shard's output state (public).
	pub fn initial_state(&self) -> Option<&StateMatrix<PackedLane8>> {
		self.initial_state.as_ref()
	}

	/// Populate a sharded sponge whose absorb chain starts from `init_state` (25 lanes) rather than zero.
	/// Also fills the injected initial-state columns (track 0 = `init_state`).
	pub fn populate_with_initial_state<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		blocks: &[StateMatrix<u64>],
		init_state: &[u64; STATE_LANES],
	) -> Result<()>
	where
		P: PackedFieldIndexable + PackedExtension<B1> + PackedExtension<B8>,
		P::Scalar: TowerField + Pod + ExtensionField<B1> + ExtensionField<B8>,
	{
		assert_eq!(blocks.len(), self.perms.len(), "one message block per permutation");
		let init_cols = self.initial_state.as_ref().expect("sharded sponge has initial_state columns");
		// Fill the injected initial-state columns (track 0 = init_state).
		for x in 0..5 {
			for y in 0..5 {
				let lane = x + 5 * y;
				let mut cell = index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(init_cols[(x, y)])?;
				cell[0] = init_state[lane];
				for t in 1..TRACKS { cell[t] = 0; }
			}
		}
		let mut running = *init_state;
		for b in 0..self.perms.len() {
			let blk = blocks[b].as_inner();
			let mut input = running;
			for i in 0..self.rate_lanes { input[i] ^= blk[i]; }
			let input_sm = StateMatrix::from_values(input);
			self.perms[b].populate_state_in(index, std::slice::from_ref(&input_sm))?;
			self.perms[b].populate(index)?;
			for x in 0..5 {
				for y in 0..5 {
					let lane = x + 5 * y;
					let mut cell = index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(self.message[b][(x, y)])?;
					cell[0] = if lane < self.rate_lanes { blk[lane] } else { 0 };
					for t in 1..TRACKS { cell[t] = 0; }
				}
			}
			if b >= 1 {
				let prev_shift = &self.prev_shifts[b - 1];
				for x in 0..5 {
					for y in 0..5 {
						let lane = x + 5 * y;
						let mut cell = index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(prev_shift[(x, y)])?;
						cell[0] = running[lane];
						for t in 1..TRACKS { cell[t] = 0; }
					}
				}
			}
			let out = self.perms[b].read_state_outs(index)?.next().expect("one row");
			running = *out.as_inner();
		}
		{
			let mut mask = index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(self.track0_mask)?;
			mask[0] = u64::MAX;
			for t in 1..TRACKS { mask[t] = 0; }
		}
		Ok(())
	}

	/// The per-block message columns. Block `b`'s block lives in track 0 (the input track) of each
	/// lane; bind these to the message source and/or read them back.
	pub fn message(&self) -> &[StateMatrix<PackedLane8>] {
		&self.message
	}

	/// The sponge output state (the final permutation's output, in track 7 of each lane).
	pub fn output(&self) -> &StateMatrix<PackedLane8> {
		self.perms.last().expect("at least one block").packed_state_out()
	}

	/// Read the sponge output state (the running state after the last block) as `u64` lanes.
	pub fn read_output<P>(&self, index: &TableWitnessSegment<P>) -> Result<StateMatrix<u64>>
	where
		P: PackedFieldIndexable + PackedExtension<B1> + PackedExtension<B8>,
		P::Scalar: TowerField + Pod + ExtensionField<B1> + ExtensionField<B8>,
	{
		Ok(self
			.perms
			.last()
			.expect("at least one block")
			.read_state_outs(index)?
			.next()
			.expect("one row"))
	}

	/// Populate the sponge for a single hash from its `n_blocks` message blocks (each a full state
	/// matrix whose rate lanes carry the block and whose capacity lanes are zero). Runs the absorb
	/// chain natively through the wrapped permutations and fills every column: the permutation
	/// inputs and round pipelines, the message columns, the aligned previous-output columns, and
	/// the track-0 selector.
	pub fn populate<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		blocks: &[StateMatrix<u64>],
	) -> Result<()>
	where
		P: PackedFieldIndexable + PackedExtension<B1> + PackedExtension<B8>,
		P::Scalar: TowerField + Pod + ExtensionField<B1> + ExtensionField<B8>,
	{
		assert_eq!(blocks.len(), self.perms.len(), "one message block per permutation");

		let mut running = [0u64; STATE_LANES];
		for b in 0..self.perms.len() {
			let blk = blocks[b].as_inner();

			// Absorb block b into the running state (rate lanes only) to form the permutation input.
			let mut input = running;
			for i in 0..self.rate_lanes {
				input[i] ^= blk[i];
			}
			let input_sm = StateMatrix::from_values(input);
			self.perms[b].populate_state_in(index, std::slice::from_ref(&input_sm))?;
			self.perms[b].populate(index)?;

			// Message column: block b in track 0 (rate lanes), zero elsewhere.
			for x in 0..5 {
				for y in 0..5 {
					let lane = x + 5 * y;
					let mut cell = index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(self.message[b][(x, y)])?;
					cell[0] = if lane < self.rate_lanes { blk[lane] } else { 0 };
					for t in 1..TRACKS {
						cell[t] = 0;
					}
				}
			}

			// Aligned previous output (block b's absorb uses the running state = prev output).
			if b >= 1 {
				let prev_shift = &self.prev_shifts[b - 1];
				for x in 0..5 {
					for y in 0..5 {
						let lane = x + 5 * y;
						let mut cell =
							index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(prev_shift[(x, y)])?;
						cell[0] = running[lane];
						for t in 1..TRACKS {
							cell[t] = 0;
						}
					}
				}
			}

			// Advance the running state to this permutation's output.
			let out = self.perms[b].read_state_outs(index)?.next().expect("one row");
			running = *out.as_inner();
		}

		// Track-0 selector.
		{
			let mut mask = index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(self.track0_mask)?;
			mask[0] = u64::MAX;
			for t in 1..TRACKS {
				mask[t] = 0;
			}
		}

		Ok(())
	}

	/// Populate a **batch** of `hashes` — one independent sound hash per table row — to amortize the
	/// fixed proving cost over the batch. `hashes[i]` is hash `i`'s `n_blocks` message blocks. Each
	/// row runs the same constrained absorb chain across the wrapped permutations; the chain is
	/// advanced using the gadget's own permutation outputs (no external Keccak). The table must have
	/// been initialised with at least `hashes.len()` rows.
	pub fn populate_batch<P>(
		&self,
		index: &mut TableWitnessSegment<P>,
		hashes: &[Vec<StateMatrix<u64>>],
	) -> Result<()>
	where
		P: PackedFieldIndexable + PackedExtension<B1> + PackedExtension<B8>,
		P::Scalar: TowerField + Pod + ExtensionField<B1> + ExtensionField<B8>,
	{
		let n = hashes.len();
		let nb = self.perms.len();
		assert!(hashes.iter().all(|h| h.len() == nb), "each hash needs n_blocks message blocks");

		let mut running: Vec<[u64; STATE_LANES]> = vec![[0u64; STATE_LANES]; n]; // per-hash running state
		for b in 0..nb {
			// Absorb block b into each hash's running state to form its permutation input (row i).
			let inputs: Vec<StateMatrix<u64>> = (0..n)
				.map(|i| {
					let blk = hashes[i][b].as_inner();
					let mut input = running[i];
					for l in 0..self.rate_lanes {
						input[l] ^= blk[l];
					}
					StateMatrix::from_values(input)
				})
				.collect();
			self.perms[b].populate_state_in(index, inputs.iter())?;
			self.perms[b].populate(index)?;

			// Message columns (block b, row i in track 0) and aligned previous outputs (row i).
			for i in 0..n {
				let blk = hashes[i][b].as_inner();
				for x in 0..5 {
					for y in 0..5 {
						let lane = x + 5 * y;
						let mut cell =
							index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(self.message[b][(x, y)])?;
						cell[TRACKS * i] = if lane < self.rate_lanes { blk[lane] } else { 0 };
						for t in 1..TRACKS {
							cell[TRACKS * i + t] = 0;
						}
					}
				}
			}
			if b >= 1 {
				let prev_shift = &self.prev_shifts[b - 1];
				for i in 0..n {
					for x in 0..5 {
						for y in 0..5 {
							let lane = x + 5 * y;
							let mut cell =
								index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(prev_shift[(x, y)])?;
							cell[TRACKS * i] = running[i][lane];
							for t in 1..TRACKS {
								cell[TRACKS * i + t] = 0;
							}
						}
					}
				}
			}

			// Advance each hash's running state from this permutation's output (row i).
			let outs: Vec<StateMatrix<u64>> = self.perms[b].read_state_outs(index)?.collect();
			for i in 0..n {
				running[i] = *outs[i].as_inner();
			}
		}

		// Track-0 selector (a constant column, filled on every row like the gadget's own link_sel).
		let mut mask = index.get_mut_as::<u64, B1, { LANE_BITS * TRACKS }>(self.track0_mask)?;
		for chunk in mask.chunks_exact_mut(TRACKS) {
			chunk[0] = u64::MAX;
			for t in 1..TRACKS {
				chunk[t] = 0;
			}
		}
		Ok(())
	}
}
