# VLA-and-VLM-PyTorch-Model-Integration-with-MATLAB-Code-Generation-
Integrating PyTorch VLA/VLM models into MATLAB to enable automated code generation for high-performance deployment.

# VLM to MATLAB R2026a Codegen Comparison Report

Date: 2026-05-15

## Executive Summary

This report compares three model paths tested in this workspace:

1. InternVL2.5 vision/projector
2. Qwen2.5-VL-3B
3. SmolVLA

The only path that has completed the full MATLAB R2026a chain so far is the
InternVL2.5 vision/projector submodule:

```text
PyTorch module -> torch.export .pt2 -> MATLAB inference -> C++ codegen -> MEX
```

The complete chat/generation VLM path is still blocked for Qwen2.5-VL and
InternVL2.5 decoder because the Hugging Face Transformers decoder path contains
dynamic causal-mask logic that `torch.export` cannot trace cleanly. SmolVLA
exports to `.pt2`, but MATLAB rejects the graph because it contains unsupported
operations.

## Result Matrix

| Model / Module | Python Inference | `.pt2` Export | MATLAB Load/Run | C++ Codegen | MEX | Current Status |
|---|---:|---:|---:|---:|---:|---|
| InternVL2.5 vision/projector | Yes | Yes | Yes | Yes | Yes | Successful submodule pipeline |
| InternVL2.5 full VLM forward | Yes | No | Not reached | Yes | Not reached | Blocked at Qwen2 decoder causal mask tracing |
| Qwen2.5-VL-3B full VLM | Yes | No | Not reached | Yes | Not reached | Blocked at vision window indexing and causal mask tracing |
| SmolVLA | Yes | Yes | Yes | Not reached | Not reached | MATLAB rejects unsupported ops |

## 1. InternVL2.5 Vision/Projector

### Target

- Model: `OpenGVLab/InternVL2_5-1B`
- Folder: `internvl25/`
- Local model directory: `internvl25/models/InternVL2_5-1B`
- Repository size: 1,887,041,469 bytes

### What Was Converted

The converted part is:

```text
image tensor -> vision encoder -> projector -> visual tokens
```

Input:

```text
1 x 3 x 448 x 448 single
```

Output:

```text
1 x 256 x 896 single
```

This module is the "eyes plus adapter" of InternVL2.5. It does not generate text
by itself. It converts an image into 256 visual tokens that are compatible with
the language model side.

### Verified Commands

Export `.pt2`:

```powershell
.\.venv_pt28\Scripts\python.exe internvl25\python\capture_internvl25_export_inputs.py
.\.venv_pt28\Scripts\python.exe internvl25\python\export_internvl25_vision_to_pt2.py
```

MATLAB direct run:

```matlab
cd("C:\Users\Fred\Documents\New project")
addpath("internvl25/matlab")
runInternVL25VisionDemo
```

C++ static library:

```matlab
generateInternVL25VisionCppLib
```

MEX:

```matlab
generateInternVL25VisionMex
```

### Artifacts

```text
internvl25/test_data/internvl25_vision.pt2
codegen/lib/internvl25VisionPredict/internvl25VisionPredict.lib
codegen/mex/internvl25VisionPredict/internvl25VisionPredict_mex.mexw64
```

### Key Fix

The first `.pt2` included `aten.upsample_bicubic2d.vec`, caused by positional
embedding interpolation. Since the experiment uses a fixed 448 x 448 input, the
export wrapper bypasses interpolation and returns the existing positional
embedding directly. After that, MATLAB could run the model.

### Validation

MATLAB direct inference succeeded.

MEX generation succeeded after setting:

```matlab
cfg.TargetLang = "C++";
```

MATLAB vs MEX maximum absolute error:

```text
0
```

## 2. Qwen2.5-VL-3B

### Target

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Local model directory: `models/qwen2_5_vl_3b_instruct`
- Repository size: 7,520,919,614 bytes

### What Works

Python backend inference works:

```powershell
.\.venv\Scripts\python.exe run_qwen25vl_local.py --prompt "Describe this image in one short sentence." --max-new-tokens 24
```

The MATLAB app can call the Python backend:

```matlab
Qwen25VLApp
```

### What Fails

The full Qwen2.5-VL fixed-shape logits wrapper runs in eager mode and returns:

```text
1 x 151936 logits
```

But `torch.export.export` fails before a `.pt2` file is produced.

Vision path failure:

```text
get_window_index -> cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
GuardOnDataDependentSymNode
```

Text-only decoder failure:

```text
create_causal_mask -> eager_mask -> sdpa_mask_recent_torch -> vmap
AssertionError: Current active mode ProxyTorchDispatchMode not registered
```

### Meaning

Qwen2.5-VL is usable as a Python backend, but the current Transformers
implementation is not export-friendly enough for the direct `.pt2` path. Since
no `.pt2` is produced, MATLAB import, C++ codegen, and MEX are not reached.

### Recommended Next Step

Do not attempt full `generate()` export. Try smaller wrappers:

- vision patch embedding only
- projector only
- decoder block with explicit precomputed causal mask
- patched visual window indexing without `.tolist()` over symbolic tensors

## 3. SmolVLA

### Target

- Model: `lerobot/smolvla_base`
- Local model directory: `models/smolvla_base`
- Exported file: `smolvla_policy.pt2`

### What Works

The project can export the SmolVLA wrapper with PyTorch 2.8.0:

```powershell
.\.venv_pt28\Scripts\python.exe export_smolvla_to_pt2.py
```

The generated file exists:

```text
smolvla_policy.pt2
```

### What Fails

MATLAB R2026a rejects the exported graph:

```matlab
loadPyTorchExportedProgram("smolvla_policy.pt2")
```

Error:

```text
Unable to load model from "smolvla_policy.pt2" as it contains unsupported operations.
```

### Meaning

SmolVLA passes the PyTorch export stage, but fails at the MATLAB operator-support
stage. This is different from Qwen2.5-VL, which fails before `.pt2` generation.

### Recommended Next Step

List unsupported or risky ops and isolate smaller modules:

```powershell
.\.venv_pt28\Scripts\python.exe list_pt2_ops.py smolvla_policy.pt2
```

Then try exporting:

- image encoder only
- action head only
- fixed MLP blocks
- smaller policy subgraphs without unsupported control flow or custom ops

## Comparison of Failure Stages

| Model | Failure Stage | Interpretation |
|---|---|---|
| Qwen2.5-VL-3B | PyTorch `torch.export` | Model code has dynamic Python/tensor interactions that block `.pt2` creation. |
| SmolVLA | MATLAB import | `.pt2` exists, but MATLAB does not support all operators in the graph. |
| InternVL2.5 full VLM | PyTorch `torch.export` | Decoder mask path blocks full VLM export. |
| InternVL2.5 vision/projector | None for tested path | Full tested submodule path succeeded. |

## Practical Recommendation

Use a two-track architecture:

### Track A: Production MATLAB App

Use Python backend for full chat models:

- Qwen2.5-VL app
- InternVL2.5 app
- future Molmo or MiniCPM backends

This gives real VLM output now.

### Track B: Codegen/MEX Acceleration

Move only export-friendly submodules into MATLAB Coder:

- InternVL2.5 vision/projector: already successful
- future image encoders
- projectors
- fixed MLP/action heads
- small fixed-shape decoder blocks if causal masks are manually controlled

This avoids blocking the project on full autoregressive generation export.

## Current Best Result

The best verified result is:

```text
InternVL2.5 vision/projector
-> internvl25_vision.pt2
-> MATLAB direct inference
-> C++ static library
-> MEX
-> MATLAB vs MEX max error = 0
```

This proves that the R2026a PyTorch ExportedProgram workflow can work for a real
VLM submodule when the exported graph avoids unsupported dynamic logic and
unsupported MATLAB operators.
