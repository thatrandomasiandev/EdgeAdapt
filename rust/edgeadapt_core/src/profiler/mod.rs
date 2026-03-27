//! Device profiling: Linux sysfs/proc readers and `DeviceState` types.

pub mod linux;
pub mod types;

use pyo3::prelude::*;

/// Registers profiler symbols on the parent PyO3 module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    linux::register(parent)
}
