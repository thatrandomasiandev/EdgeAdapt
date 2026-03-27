//! PyO3 extension for edgeadapt: device profiling and hot-swap primitives.
#![allow(clippy::useless_conversion)] // False positives with PyO3 `PyResult` and `?`.

pub mod profiler;
pub mod swapper;
mod utils;

use pyo3::prelude::*;

/// Initializes the `edgeadapt._edgeadapt_core` Python module.
#[pymodule]
fn _edgeadapt_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    profiler::register(m)?;
    swapper::register(m)?;
    Ok(())
}
