// Copyright 2024-2025 Irreducible Inc.
// Modified 2025 by Anonymous (Apache-2.0 derivative): generalized the ring-switch
// tensor-algebra kappa dispatch from the hardcoded tower-level-7 literals to
// type-derived guards, so the challenge/extension field may sit at any tower level.

use binius_field::tower::{PackedTop, TowerFamily};

use super::error::Error;
use crate::tensor_algebra::TensorAlgebra;

type FExt<Tower> = <Tower as TowerFamily>::B128;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TowerTensorAlgebra<Tower: TowerFamily> {
	B1(TensorAlgebra<Tower::B1, Tower::B128>),
	B8(TensorAlgebra<Tower::B8, Tower::B128>),
	B16(TensorAlgebra<Tower::B16, Tower::B128>),
	B32(TensorAlgebra<Tower::B32, Tower::B128>),
	B64(TensorAlgebra<Tower::B64, Tower::B128>),
	B128(TensorAlgebra<Tower::B128, Tower::B128>),
}

impl<Tower: TowerFamily> TowerTensorAlgebra<Tower> {
	/// Constructs an element from a vector of vertical subring elements.
	///
	/// ## Preconditions
	///
	/// * `elems` must have length `FE::DEGREE`, otherwise this will pad or truncate.
	pub fn new(kappa: usize, elems: Vec<FExt<Tower>>) -> Result<Self, Error> {
		// The dispatch key is $\kappa = \log_2[\FExt : \text{subfield}]$, which shifts
		// with `FExt`'s tower level. Rather than hardcode the level-7 (128-bit FExt)
		// literals {7,4,3,2,1,0}, derive each arm's $\kappa$ from the type, so a
		// level-8 `FExt` (e.g. a 256-bit B128 slot) auto-yields {8,5,4,3,2,0} etc.
		match kappa {
			k if k == TensorAlgebra::<Tower::B1, FExt<Tower>>::kappa() => {
				Ok(Self::B1(TensorAlgebra::new(elems)))
			}
			k if k == TensorAlgebra::<Tower::B8, FExt<Tower>>::kappa() => {
				Ok(Self::B8(TensorAlgebra::new(elems)))
			}
			k if k == TensorAlgebra::<Tower::B16, FExt<Tower>>::kappa() => {
				Ok(Self::B16(TensorAlgebra::new(elems)))
			}
			k if k == TensorAlgebra::<Tower::B32, FExt<Tower>>::kappa() => {
				Ok(Self::B32(TensorAlgebra::new(elems)))
			}
			k if k == TensorAlgebra::<Tower::B64, FExt<Tower>>::kappa() => {
				Ok(Self::B64(TensorAlgebra::new(elems)))
			}
			k if k == TensorAlgebra::<Tower::B128, FExt<Tower>>::kappa() => {
				Ok(Self::B128(TensorAlgebra::new(elems)))
			}
			_ => Err(Error::PackingDegreeNotSupported { kappa }),
		}
	}

	/// Returns the additive identity element, zero.
	pub fn zero(kappa: usize) -> Result<Self, Error> {
		match kappa {
			k if k == TensorAlgebra::<Tower::B1, FExt<Tower>>::kappa() => {
				Ok(Self::B1(TensorAlgebra::default()))
			}
			k if k == TensorAlgebra::<Tower::B8, FExt<Tower>>::kappa() => {
				Ok(Self::B8(TensorAlgebra::default()))
			}
			k if k == TensorAlgebra::<Tower::B16, FExt<Tower>>::kappa() => {
				Ok(Self::B16(TensorAlgebra::default()))
			}
			k if k == TensorAlgebra::<Tower::B32, FExt<Tower>>::kappa() => {
				Ok(Self::B32(TensorAlgebra::default()))
			}
			k if k == TensorAlgebra::<Tower::B64, FExt<Tower>>::kappa() => {
				Ok(Self::B64(TensorAlgebra::default()))
			}
			k if k == TensorAlgebra::<Tower::B128, FExt<Tower>>::kappa() => {
				Ok(Self::B128(TensorAlgebra::default()))
			}
			_ => Err(Error::PackingDegreeNotSupported { kappa }),
		}
	}

	/// Returns $\kappa$, the base-2 logarithm of the extension degree.
	///
	/// Derived per-variant from the type so it tracks `FExt`'s tower level (level 7
	/// gives {7,4,3,2,1,0}; level 8 gives {8,5,4,3,2,0}).
	pub fn kappa(&self) -> usize {
		match self {
			Self::B1(_) => TensorAlgebra::<Tower::B1, FExt<Tower>>::kappa(),
			Self::B8(_) => TensorAlgebra::<Tower::B8, FExt<Tower>>::kappa(),
			Self::B16(_) => TensorAlgebra::<Tower::B16, FExt<Tower>>::kappa(),
			Self::B32(_) => TensorAlgebra::<Tower::B32, FExt<Tower>>::kappa(),
			Self::B64(_) => TensorAlgebra::<Tower::B64, FExt<Tower>>::kappa(),
			Self::B128(_) => TensorAlgebra::<Tower::B128, FExt<Tower>>::kappa(),
		}
	}

	/// Returns a slice of the vertical subfield elements composing the tensor algebra element.
	pub fn vertical_elems(&self) -> &[FExt<Tower>] {
		match self {
			Self::B1(elem) => elem.vertical_elems(),
			Self::B8(elem) => elem.vertical_elems(),
			Self::B16(elem) => elem.vertical_elems(),
			Self::B32(elem) => elem.vertical_elems(),
			Self::B64(elem) => elem.vertical_elems(),
			Self::B128(elem) => elem.vertical_elems(),
		}
	}

	/// Multiply by an element from the vertical subring.
	pub fn scale_vertical(self, scalar: FExt<Tower>) -> Self {
		match self {
			Self::B1(elem) => Self::B1(elem.scale_vertical(scalar)),
			Self::B8(elem) => Self::B8(elem.scale_vertical(scalar)),
			Self::B16(elem) => Self::B16(elem.scale_vertical(scalar)),
			Self::B32(elem) => Self::B32(elem.scale_vertical(scalar)),
			Self::B64(elem) => Self::B64(elem.scale_vertical(scalar)),
			Self::B128(elem) => Self::B128(elem.scale_vertical(scalar)),
		}
	}

	/// Adds the right hand size into the current value.
	///
	/// ## Throws
	///
	/// * [`Error::TowerLevelMismatch`] if the arguments' underlying tower level do not match
	pub fn add_assign(&mut self, rhs: &Self) -> Result<(), Error> {
		match (self, rhs) {
			(Self::B1(lhs), Self::B1(rhs)) => *lhs += rhs,
			(Self::B8(lhs), Self::B8(rhs)) => *lhs += rhs,
			(Self::B16(lhs), Self::B16(rhs)) => *lhs += rhs,
			(Self::B32(lhs), Self::B32(rhs)) => *lhs += rhs,
			(Self::B64(lhs), Self::B64(rhs)) => *lhs += rhs,
			(Self::B128(lhs), Self::B128(rhs)) => *lhs += rhs,
			_ => return Err(Error::TowerLevelMismatch),
		}
		Ok(())
	}
}

impl<Tower> TowerTensorAlgebra<Tower>
where
	Tower: TowerFamily,
	FExt<Tower>: PackedTop<Tower>,
{
	/// Fold the tensor algebra element into a field element by scaling the rows and accumulating.
	///
	/// ## Preconditions
	///
	/// * `coeffs` must have length $2^\kappa$
	pub fn fold_vertical(self, coeffs: &[FExt<Tower>]) -> FExt<Tower> {
		match self {
			Self::B1(elem) => elem.fold_vertical(coeffs),
			Self::B8(elem) => elem.fold_vertical(coeffs),
			Self::B16(elem) => elem.fold_vertical(coeffs),
			Self::B32(elem) => elem.fold_vertical(coeffs),
			Self::B64(elem) => elem.fold_vertical(coeffs),
			Self::B128(elem) => elem.fold_vertical(coeffs),
		}
	}
}
