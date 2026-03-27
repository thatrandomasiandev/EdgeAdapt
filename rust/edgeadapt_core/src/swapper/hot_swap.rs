//! Background load + atomic swap of a Python-owned backend object.

use pyo3::prelude::*;
use std::sync::{Arc, RwLock};

/// Coordinates variant swaps with a Python ``loader(variant_name)`` callable.
#[pyclass(name = "HotSwapper")]
pub struct HotSwapper {
    loader: PyObject,
    current_variant: Arc<RwLock<String>>,
    active_backend: Arc<RwLock<Option<PyObject>>>,
    on_swap_start: Option<PyObject>,
    on_swap_complete: Option<PyObject>,
    on_fallback: Option<PyObject>,
}

#[pymethods]
impl HotSwapper {
    #[new]
    #[pyo3(signature = (loader, initial_variant, on_swap_start=None, on_swap_complete=None, on_fallback=None))]
    #[allow(clippy::too_many_arguments)]
    fn py_new(
        loader: PyObject,
        initial_variant: String,
        on_swap_start: Option<PyObject>,
        on_swap_complete: Option<PyObject>,
        on_fallback: Option<PyObject>,
    ) -> Self {
        Self {
            loader,
            current_variant: Arc::new(RwLock::new(initial_variant)),
            active_backend: Arc::new(RwLock::new(None)),
            on_swap_start,
            on_swap_complete,
            on_fallback,
        }
    }

    #[getter]
    fn active_variant(&self) -> PyResult<String> {
        let g = self.current_variant.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {e}"))
        })?;
        Ok(g.clone())
    }

    fn load_initial(&self, py: Python<'_>, variant: String) -> PyResult<()> {
        let fun = self.loader.bind(py);
        let obj = fun.call1((variant.as_str(),))?;
        let py_obj = obj.into_py(py);
        if let Ok(mut g) = self.active_backend.write() {
            *g = Some(py_obj);
        }
        if let Ok(mut av) = self.current_variant.write() {
            *av = variant;
        }
        Ok(())
    }

    fn get_backend(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        let g = self.active_backend.read().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("lock poisoned: {e}"))
        })?;
        Ok(g.as_ref().map(|p| p.clone_ref(py)))
    }

    fn swap_to(&self, py: Python<'_>, variant: String) -> PyResult<()> {
        let loader = self.loader.clone_ref(py);
        let current_variant = self.current_variant.clone();
        let active_backend = self.active_backend.clone();
        let on_start = self.on_swap_start.as_ref().map(|c| c.clone_ref(py));
        let on_complete = self.on_swap_complete.as_ref().map(|c| c.clone_ref(py));
        let on_fallback = self.on_fallback.as_ref().map(|c| c.clone_ref(py));

        py.allow_threads(move || {
            std::thread::spawn(move || {
                Python::with_gil(|py| {
                    if let Some(cb) = &on_start {
                        let _ = cb.call1(py, (variant.as_str(),));
                    }

                    let fun = loader.bind(py);
                    match fun.call1((variant.as_str(),)) {
                        Ok(obj) => {
                            let py_obj = obj.into_py(py);
                            if let Ok(mut g) = active_backend.write() {
                                *g = Some(py_obj);
                            }
                            if let Ok(mut av) = current_variant.write() {
                                *av = variant.clone();
                            }
                            if let Some(cb) = &on_complete {
                                let _ = cb.call1(py, (variant.as_str(),));
                            }
                        }
                        Err(err) => {
                            // Do not mutate `active_backend` or `current_variant` — last-known-good serves.
                            if let Some(cb) = &on_fallback {
                                let msg = format!("{err}");
                                let _ = cb.call1(py, (variant.as_str(), msg));
                            }
                        }
                    }
                });
            });
        });
        Ok(())
    }
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<HotSwapper>()?;
    Ok(())
}
