"""What this computer offers the solvers: GPU, Gurobi, CPU and memory.

    import solver.machine as machine
    machine.setup()                      # detect, check, save ~/.rum/machine.json; run once after install
    machine.setup(use_gpu=False)         # override: never use the GPU
    profile = machine.load()             # the saved profile (detected now if there is none)

GPU support: NVIDIA only, through CuPy. CUDA 13 needs compute capability >= 7.5 (Turing and
newer); CUDA 12 also runs Maxwell, Pascal and Volta (>= 5.0); RTX 50xx (12.0) needs CUDA >= 12.8.
Tested: RTX 5090 (32 GB) and H200 (141 GB). The kernels use plain CUDA (no tensor cores, half
precision or atomics), so any supported card works; memory sets the largest n.

Detection reads the card through nvidia-smi and the CuPy package version only, so it does not
create a CUDA context. Free GPU memory is read live (gpu_free_bytes) just before a GPU method runs.
"""
import json
import os
import platform
import subprocess
import time

PROFILE_PATH = os.path.join(os.path.expanduser("~"), ".rum", "machine.json")


def _gpu():
    """First NVIDIA GPU as seen by nvidia-smi, plus the CuPy build; None without either."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        name, mem_mib, cc, driver = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
    except Exception:
        return None
    gpu = dict(name=name, memory_total=int(mem_mib) * 2 ** 20, compute_capability=cc, driver=driver,
               cupy=None, cuda_runtime=None)
    try:
        import cupy
        gpu["cupy"] = cupy.__version__
        v = int(cupy.cuda.runtime.runtimeGetVersion())          # e.g. 13030 -> "13.3"
        gpu["cuda_runtime"] = "%d.%d" % (v // 1000, (v % 1000) // 10)
    except Exception:
        pass
    gpu["supported"], gpu["reason"] = gpu_supported(gpu)
    return gpu


def gpu_supported(gpu):
    """(ok, reason) for a GPU entry of the profile."""
    if gpu is None:
        return False, "no NVIDIA GPU found (nvidia-smi)"
    if not gpu.get("cupy") or not gpu.get("cuda_runtime"):
        return False, "CuPy is not installed or cannot load the CUDA runtime"
    cc = float(gpu["compute_capability"])
    cuda = tuple(int(x) for x in gpu["cuda_runtime"].split("."))
    if cuda[0] >= 13 and cc < 7.5:
        return False, "CUDA %s needs compute capability >= 7.5; this card has %s" % (gpu["cuda_runtime"], cc)
    if cc < 5.0:
        return False, "compute capability %s is too old (>= 5.0 needed)" % cc
    if cc >= 12.0 and cuda < (12, 8):
        return False, "RTX 50xx (compute capability 12.0) needs CUDA >= 12.8; CuPy uses %s" % gpu["cuda_runtime"]
    return True, "%s, %.0f GB, compute capability %s, CUDA %s" % (
        gpu["name"], gpu["memory_total"] / 2 ** 30, cc, gpu["cuda_runtime"])


def _gurobi(check_licence):
    try:
        import gurobipy as gp
        g = dict(version="%d.%d.%d" % gp.gurobi.version(), licence=None)
    except Exception:
        return None
    if check_licence:
        try:
            from .common.gurobi_env import _ENV       # the env every Gurobi solver uses
            _ENV.getParam("Threads")
            g["licence"] = "ok"
        except Exception as e:
            g["licence"] = "error: %s" % e
    return g


def _ram_bytes():
    try:
        if os.name == "nt":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MS()
            ms.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return int(ms.ullTotalPhys)
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return None


def detect(check_licence=False):
    """The profile of this machine (nothing is saved)."""
    try:
        import numba
        nb = numba.__version__
    except Exception:
        nb = None
    return dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node(),
                platform=platform.platform(), python=platform.python_version(),
                cpu=platform.processor(), cpu_count=os.cpu_count(), ram_bytes=_ram_bytes(),
                numba=nb, gpu=_gpu(), gurobi=_gurobi(check_licence), use_gpu=True, use_gurobi=True)


def setup(use_gpu=None, use_gurobi=None, self_test=False, path=PROFILE_PATH, verbose=True):
    """Detect the machine, apply overrides, optionally self-test, and save the profile.
    use_gpu / use_gurobi = False keeps the RUM class off that resource even when present.
    self_test=True compiles the GPU kernels and solves a tiny instance with each available method."""
    prof = detect(check_licence=True)
    if use_gpu is not None:
        prof["use_gpu"] = bool(use_gpu)
    if use_gurobi is not None:
        prof["use_gurobi"] = bool(use_gurobi)
    if self_test:
        prof["self_test"] = _self_test(prof)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(prof, f, indent=2)
    if verbose:
        print(summary(prof))
        print("saved", path)
    return prof


_DETECTED = {}


def load(path=PROFILE_PATH):
    """The saved profile; detected once per process (not saved) when setup() has not been run."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        if "prof" not in _DETECTED:
            _DETECTED["prof"] = detect()
        return _DETECTED["prof"]


def can_use_gpu(prof):
    return bool(prof.get("use_gpu", True) and prof.get("gpu") and prof["gpu"].get("supported"))


def can_use_gurobi(prof):
    g = prof.get("gurobi")
    return bool(prof.get("use_gurobi", True) and g and g.get("licence") in (None, "ok"))


def gpu_free_bytes():
    """Free device memory now (creates the CUDA context if there is none yet)."""
    import cupy as cp
    return int(cp.cuda.runtime.memGetInfo()[0])


def summary(prof):
    g, gu = prof.get("gpu"), prof.get("gurobi")
    lines = ["machine: %s, %s cores, %.0f GB RAM, Python %s, numba %s"
             % (prof.get("cpu") or prof.get("platform"), prof.get("cpu_count"),
                (prof.get("ram_bytes") or 0) / 2 ** 30, prof.get("python"), prof.get("numba"))]
    lines.append("GPU:     " + (("%s -> %s" % (g["reason"], "used" if can_use_gpu(prof) else "NOT used"))
                                if g else "none"))
    lines.append("Gurobi:  " + (("%s, licence %s -> %s" % (gu["version"], gu.get("licence") or "not checked",
                                                          "used" if can_use_gurobi(prof) else "NOT used"))
                                if gu else "not installed"))
    if prof.get("self_test"):
        lines.append("self-test: " + ", ".join("%s %s" % kv for kv in prof["self_test"].items()))
    return "\n".join(lines)


def _self_test(prof):
    """Tiny certified solve with each method the profile allows."""
    import numpy as np
    from .rum_solver import RUM
    from .common.instances import warmup_instance
    cd = warmup_instance(6)
    rum = RUM(cd[:, 0].astype(np.int64), cd[:, 1].astype(np.int64), cd[:, 2])
    methods = (["bm_admm", "flow_cg", "flow_admm"] if can_use_gpu(prof) else []) + \
              (["bm_qp", "ip_cg_heur"] if can_use_gurobi(prof) else [])
    out = {}
    for m in methods:
        try:
            r = rum.calculate_distance(method=m, max_time=120, machine=prof)
            out[m] = "ok" if r.exit_flag > 0 else "exit_flag %d" % r.exit_flag
        except Exception as e:
            out[m] = "error: %s" % e
    return out
