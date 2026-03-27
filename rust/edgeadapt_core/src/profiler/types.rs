//! Normalized device state types (mirrors Python `DeviceState`).

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

/// Snapshot of device constraints used for policy decisions.
#[derive(Debug, Clone, Default)]
pub struct DeviceState {
    pub battery_pct: Option<f64>,
    pub is_charging: Option<bool>,
    pub thermal_zone_c: Option<f64>,
    pub available_ram_mb: f64,
    pub cpu_utilization_pct: f64,
    pub gpu_utilization_pct: Option<f64>,
    pub network_quality: Option<String>,
    pub timestamp: f64,
}

impl DeviceState {
    /// Converts this state into a Python dict compatible with `edgeadapt.profiler.base.DeviceState`.
    pub fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new_bound(py);
        match self.battery_pct {
            Some(v) => d.set_item("battery_pct", v)?,
            None => d.set_item("battery_pct", py.None())?,
        }
        match self.is_charging {
            Some(v) => d.set_item("is_charging", v)?,
            None => d.set_item("is_charging", py.None())?,
        }
        match self.thermal_zone_c {
            Some(v) => d.set_item("thermal_zone_c", v)?,
            None => d.set_item("thermal_zone_c", py.None())?,
        }
        d.set_item("available_ram_mb", self.available_ram_mb)?;
        d.set_item("cpu_utilization_pct", self.cpu_utilization_pct)?;
        match self.gpu_utilization_pct {
            Some(v) => d.set_item("gpu_utilization_pct", v)?,
            None => d.set_item("gpu_utilization_pct", py.None())?,
        }
        match &self.network_quality {
            Some(v) => d.set_item("network_quality", v.as_str())?,
            None => d.set_item("network_quality", py.None())?,
        }
        d.set_item("timestamp", self.timestamp)?;
        Ok(d)
    }
}

/// Reads a small text file and trims whitespace; returns `None` if missing or unreadable.
pub fn read_trimmed_string(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Parses first line as f64.
pub fn parse_first_float(text: &str) -> Option<f64> {
    text.lines().next()?.trim().parse().ok()
}
