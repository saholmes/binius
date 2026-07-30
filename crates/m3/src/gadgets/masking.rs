// Copyright 2026 The pq-rollup Authors
// Addition to the Binius fork (Apache-2.0). See MODIFICATIONS.md.
//
// A4 (zero-knowledge roadmap): **bounded-independence masking columns**.
//
// `MaskColumns` adds `count` committed, UNCONSTRAINED columns to a table and fills them with fresh
// randomness. They are the *randomness layer* for a zero-knowledge argument over binary tower
// fields: on their own they hide nothing about the witness (a witness column's openings still reveal
// its values), but they are the material a masked polynomial commitment (A2) and a zero-knowledge
// sumcheck (A3) combine into the opened evaluations so those become independent of the witness.
// Because the mask columns satisfy no constraint, any random fill validates; choose `count` larger
// than the number of opened evaluations to obtain bounded independence.
//
// This is entirely opt-in: a table only gains masking if it constructs a `MaskColumns`. Existing
// gadgets and tables are unaffected.

use binius_field::{ExtensionField, PackedExtension, PackedFieldIndexable, TowerField};
use bytemuck::Pod;

use crate::builder::{Col, TableBuilder, TableWitnessSegment, B1};

/// A set of committed, unconstrained masking columns of width `W` bits.
pub struct MaskColumns<const W: usize> {
	cols: Vec<Col<B1, W>>,
}

impl<const W: usize> MaskColumns<W> {
	/// Add `count` unconstrained committed mask columns of width `W` to `table`.
	pub fn new<F: TowerField>(table: &mut TableBuilder<F>, count: usize) -> Self {
		let cols = (0..count).map(|i| table.add_committed::<B1, W>(format!("zk_mask[{i}]"))).collect();
		Self { cols }
	}

	/// The mask column handles, for a masked commitment / sumcheck to combine into the witness.
	pub fn columns(&self) -> &[Col<B1, W>] {
		&self.cols
	}

	/// The number of mask columns.
	pub fn len(&self) -> usize {
		self.cols.len()
	}

	/// Whether there are no mask columns.
	pub fn is_empty(&self) -> bool {
		self.cols.is_empty()
	}

	/// Fill every mask column, on every row of the segment, with fresh randomness. `fill` writes
	/// random bytes into the buffer it is given (e.g. `|buf| rng.fill_bytes(buf)`); the gadget stays
	/// free of any RNG dependency, and the caller supplies the entropy source (the OS in production).
	pub fn populate<P>(&self, index: &mut TableWitnessSegment<P>, mut fill: impl FnMut(&mut [u8])) -> Result<(), anyhow::Error>
	where
		P: PackedExtension<B1> + PackedFieldIndexable,
		P::Scalar: TowerField + ExtensionField<B1> + Pod,
	{
		for &col in &self.cols {
			let mut cell = index.get_mut_as::<u8, B1, W>(col)?;
			fill(&mut cell[..]);
		}
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use binius_field::{arch::OptimalUnderlier128b, as_packed_field::PackedType, BinaryField128b as B128};
	use bumpalo::Bump;
	use rand::{rngs::StdRng, RngCore, SeedableRng};

	use super::MaskColumns;
	use crate::builder::{ConstraintSystem, Statement, WitnessIndex};

	#[test]
	fn mask_columns_are_unconstrained_and_random() {
		let mut cs = ConstraintSystem::new();
		let mut table = cs.add_table("MaskProbe");
		let table_id = table.id();
		let allocator = Bump::new();

		// A masked table with only masking columns: they carry no constraint.
		let mask = MaskColumns::<32>::new(&mut table, 8);

		let n_rows = 1 << 6;
		let statement = Statement { boundaries: vec![], table_sizes: vec![n_rows] };
		let mut witness = WitnessIndex::<PackedType<OptimalUnderlier128b, B128>>::new(&cs, &allocator);
		let table_witness = witness.init_table(table_id, n_rows).unwrap();
		let mut segment = table_witness.full_segment();

		let mut rng = StdRng::seed_from_u64(0xC0FFEE);
		mask.populate(&mut segment, |buf| rng.fill_bytes(buf)).unwrap();

		// The masks were actually filled with randomness (not left zero).
		let any_nonzero = mask
			.columns()
			.iter()
			.any(|&c| segment.get_as::<u8, _, 32>(c).unwrap().iter().any(|&b| b != 0));
		assert!(any_nonzero, "mask columns must be filled with randomness");

		// An unconstrained, arbitrarily-filled masked table still satisfies every constraint.
		let ccs = cs.compile(&statement).unwrap();
		let witness = witness.into_multilinear_extension_index();
		binius_core::constraint_system::validate::validate_witness(&ccs, &[], &witness).unwrap();
	}
}
