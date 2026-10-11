"""Execute the notebook from a fresh kernel and persist its outputs.

Run with `.venv/bin/python run_notebook.py` after installing requirements.txt.
"""

import os
from pathlib import Path
import sys

# Bound native threads before starting the kernel; Grid Search handles parallelism.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


class EnvironmentKernelSpecManager(KernelSpecManager):
    def get_kernel_spec(self, kernel_name):
        spec = super().get_kernel_spec("python3")
        spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        return spec


def main():
    root = Path(__file__).resolve().parent
    path = root / "challenge1.ipynb"
    notebook = nbformat.read(path, as_version=4)

    def started(cell, cell_index, **kwargs):
        print(f"Running cell {cell_index}", flush=True)

    def completed(cell, cell_index, **kwargs):
        nbformat.write(notebook, path)
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                print(output.text[-2500:], flush=True)

    manager = KernelManager(kernel_spec_manager=EnvironmentKernelSpecManager())
    client = NotebookClient(
        notebook, km=manager, timeout=3600,
        resources={"metadata": {"path": str(root)}},
        on_cell_start=started, on_cell_executed=completed,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    print("Notebook successfully executed.", flush=True)


if __name__ == "__main__":
    main()
