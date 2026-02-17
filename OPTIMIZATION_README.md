# AvaFrame Cython Optimization Guide

## What Changed

The `setup.py` now includes compiler optimizations for faster simulation performance:

### Default Behavior (Production Mode)
- **Windows (MSVC)**: `/Ox` (maximum optimization) + `/fp:fast` (fast floating-point)
- **Linux (GCC)**: `-O3` + `-ffast-math` + `-march=native`
- **Cython**: `linetrace=False` (no profiling overhead)

**Expected speedup**: ~15-40% faster simulations

### Backwards Compatibility

If you encounter compilation issues or unexpected behavior, you can disable optimizations:

```bash
# Fallback to safe defaults (no optimizations)
DISABLE_OPTIMIZATIONS=1 python setup.py build_ext --inplace
```

### Profiling Mode

To enable line-by-line profiling (reduces performance):

```bash
# Enable Cython line tracing for profiling
CYTHON_TRACE=1 python setup.py build_ext --inplace
```

## How to Recompile

### On Server (Linux)

```bash
# Activate conda environment
conda activate ALARM_env

# Navigate to AvaFrame directory
cd /path/to/ALARM_pipeline/AvaFrame

# Pull latest changes
git pull

# Clean old compiled files
rm -f avaframe/com1DFA/*.so
rm -rf build/

# Recompile with optimizations
python setup.py build_ext --inplace

# Verify compilation
python -c "import avaframe.com1DFA.DFAfunctionsCython; print('Compilation successful!')"
```

### On Windows

```powershell
# Activate conda environment
conda activate ALARM_env

# Navigate to AvaFrame directory
cd C:\path\to\ALARM_pipeline\AvaFrame

# Pull latest changes
git pull

# Clean old compiled files
Remove-Item avaframe\com1DFA\*.pyd -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

# Recompile with optimizations
python setup.py build_ext --inplace

# Verify compilation
python -c "import avaframe.com1DFA.DFAfunctionsCython; print('Compilation successful!')"
```

## Troubleshooting

### Compilation Fails with Optimization Flags

```bash
# Use safe fallback mode
DISABLE_OPTIMIZATIONS=1 python setup.py build_ext --inplace
```

### Results Look Different

The optimizations should **not** change results significantly. If you see differences:
1. Check if `-ffast-math` causes issues (floating-point precision)
2. Recompile without optimizations and compare
3. Report the issue

### Performance Not Improved

1. Verify compiled files are new: `ls -l avaframe/com1DFA/*.so`
2. Check no `DISABLE_OPTIMIZATIONS=1` was set
3. Ensure old `.pyc` files are cleared: `find . -name "*.pyc" -delete`

## Technical Details

### Compiler Flags Explained

| Flag | Effect |
|------|--------|
| `/Ox` (MSVC) | Maximum optimization (inlining, loop unrolling, etc.) |
| `-O3` (GCC) | Aggressive optimization including vectorization |
| `/fp:fast` | Relaxed floating-point precision for speed |
| `-ffast-math` | Fast math operations (may affect precision slightly) |
| `-march=native` | Use CPU-specific instructions (AVX2, SSE4, etc.) |

### Why `linetrace=False`?

`linetrace=True` adds profiling hooks to **every line** of Cython code, causing ~10-30% slowdown. It's only needed for profiling, not production.

## Version History

- **2026-02-17**: Initial optimization implementation
  - Added compiler optimization flags
  - Disabled linetrace by default
  - Added DISABLE_OPTIMIZATIONS fallback
