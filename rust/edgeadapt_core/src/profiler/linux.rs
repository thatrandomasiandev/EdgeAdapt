//! Linux-specific profiler: sysfs and `/proc` readers.

use crate::profiler::types::{parse_first_float, read_trimmed_string, DeviceState};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Previous CPU jiffies for utilization delta.
struct CpuSample {
    active: u64,
    total: u64,
}

static LAST_CPU: Mutex<Option<CpuSample>> = Mutex::new(None);

fn parse_cpu_line(line: &str) -> Option<(u64, u64)> {
    let mut parts = line.split_whitespace();
    if parts.next()? != "cpu" {
        return None;
    }
    let nums: Vec<u64> = parts.filter_map(|s| s.parse().ok()).collect();
    if nums.len() < 4 {
        return None;
    }
    let idle = nums[3] + nums.get(4).copied().unwrap_or(0);
    let total: u64 = nums.iter().sum();
    let active = total.saturating_sub(idle);
    Some((active, total))
}

fn read_cpu_utilization(proc_root: &Path) -> io::Result<f64> {
    let stat_path = proc_root.join("stat");
    let text = fs::read_to_string(&stat_path)?;
    let line = text
        .lines()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty /proc/stat"))?;
    let (active, total) = parse_cpu_line(line).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "could not parse first cpu line in /proc/stat",
        )
    })?;

    let mut guard = LAST_CPU
        .lock()
        .map_err(|e| io::Error::other(format!("cpu mutex poisoned: {e}")))?;

    let util = match *guard {
        Some(ref prev) => {
            let da = active.saturating_sub(prev.active);
            let dt = total.saturating_sub(prev.total);
            if dt == 0 {
                0.0
            } else {
                (da as f64 / dt as f64) * 100.0
            }
        }
        None => 0.0,
    };
    *guard = Some(CpuSample { active, total });
    Ok(util.clamp(0.0, 100.0))
}

fn read_mem_available_mb(proc_root: &Path) -> io::Result<f64> {
    let meminfo = fs::read_to_string(proc_root.join("meminfo"))?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: f64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            return Ok(kb / 1024.0);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "MemAvailable not found in meminfo",
    ))
}

fn read_battery(sys_root: &Path) -> io::Result<(Option<f64>, Option<bool>)> {
    let ps_root = sys_root.join("class/power_supply");
    let entries = match fs::read_dir(&ps_root) {
        Ok(e) => e,
        Err(_) => return Ok((None, None)),
    };

    let mut best_capacity: Option<f64> = None;
    let mut best_charging: Option<bool> = None;

    for ent in entries.flatten() {
        let path = ent.path();
        let type_path = path.join("type");
        let t = match read_trimmed_string(&type_path) {
            Some(t) => t,
            None => continue,
        };
        if t != "Battery" {
            continue;
        }

        if let Some(cap) = read_trimmed_string(&path.join("capacity")) {
            if let Ok(v) = cap.parse::<f64>() {
                best_capacity = Some(v.clamp(0.0, 100.0));
            }
        }

        if let Some(st) = read_trimmed_string(&path.join("status")) {
            let charging = match st.as_str() {
                "Charging" => Some(true),
                "Discharging" => Some(false),
                "Full" => Some(true),
                "Not charging" => Some(false),
                _ => None,
            };
            best_charging = charging.or(best_charging);
        }
        break;
    }

    Ok((best_capacity, best_charging))
}

fn read_max_thermal_c(sys_root: &Path) -> Option<f64> {
    let thermal_root = sys_root.join("class/thermal");
    let entries = fs::read_dir(&thermal_root).ok()?;
    let mut max_milli: Option<f64> = None;
    for ent in entries.flatten() {
        let name = ent.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("thermal_zone") {
            continue;
        }
        let temp_path = ent.path().join("temp");
        let text = fs::read_to_string(&temp_path).ok()?;
        let milli = parse_first_float(&text)?;
        max_milli = Some(match max_milli {
            Some(m) => m.max(milli),
            None => milli,
        });
    }
    max_milli.map(|m| m / 1000.0)
}

/// Reads normalized device state from a Linux-like tree rooted at `root` (defaults to `/`).
pub fn read_linux_state(root: &Path) -> io::Result<DeviceState> {
    let proc_root = root.join("proc");
    let sys_root = root.join("sys");

    let available_ram_mb = read_mem_available_mb(&proc_root)?;
    let cpu_utilization_pct = read_cpu_utilization(&proc_root)?;
    let (battery_pct, is_charging) = read_battery(&sys_root)?;
    let thermal_zone_c = read_max_thermal_c(&sys_root);

    Ok(DeviceState {
        battery_pct,
        is_charging,
        thermal_zone_c,
        available_ram_mb,
        cpu_utilization_pct,
        gpu_utilization_pct: None,
        network_quality: None,
        timestamp: 0.0,
    })
}

/// Python entry: `get_device_state(root=None)` returning a dict mirroring `DeviceState`.
#[pyfunction]
#[pyo3(signature = (root=None))]
fn get_device_state(py: Python<'_>, root: Option<PathBuf>) -> PyResult<PyObject> {
    let base = root.unwrap_or_else(|| PathBuf::from("/"));
    let state = read_linux_state(&base).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("linux profiler read failed: {e}"))
    })?;
    let d = state.to_py_dict(py)?;
    Ok(d.into())
}

/// Registers profiler Python API on the parent module.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_function(wrap_pyfunction!(get_device_state, parent)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::parse_cpu_line;

    #[test]
    fn parses_cpu_line() {
        let line = "cpu 1 2 3 4 5 6 7 8 9 10";
        let (active, total) = parse_cpu_line(line).expect("parse");
        assert!(total > 0);
        assert!(active <= total);
    }
}
