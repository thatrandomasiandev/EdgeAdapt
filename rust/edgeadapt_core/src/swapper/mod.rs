//! Hot-swap coordination (populated in Step 6).

pub mod hot_swap;

use pyo3::prelude::*;

/// Registers swapper symbols on the parent PyO3 module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    hot_swap::register(parent)
}
