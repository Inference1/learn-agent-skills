---
name: ascend-inference-repos-copilot
description: 专用于以下昇腾（Ascend）推理开源代码仓库的智能问答技能：vLLM、vLLM-Ascend、MindIE-LLM、MindIE-SD、MindIE-Motor、MindIE-PyMotor、MindIE-Turbo 以及 msModelSlim (MindStudio-ModelSlim)。当回答用户关于前述代码仓的问题时，需提供因果链感知、基于证据的技术回答，并确保回答完整、准确、逻辑合理且可追溯。覆盖的技术问题包括但不限于：源码理解、软件架构、部署与使用步骤指引、API 和参数配置、模型与特性支持查询、模型量化后如何推理、调试技巧、测试验证、故障排除与解决、日志异常检测、性能优化、精度验证、定制开发以及其他相关技术问题。支持中英文双语回复，还可以借助 DeepWiki MCP 工具，对仓库中的信息进行更深入的检索。触发条件（满足以下任意一项即可）：1. 用户的问题中提及上述任一仓库名称（支持中英文别名，且不区分大小写）；2. 用户的问题中同时包含 "昇腾 推理" 或 "Ascend Inference"，并且涉及大模型、多模态、部署、性能、报错或代码等相关信息。Use this skill whenever the user asks about vLLM on Ascend/NPU hardware, MindIE components, ModelSlim quantization, or any Huawei Ascend inference stack — even if they don't explicitly name a repository.
---

# Specialized Intelligent Q&A for Ascend Inference Open-Source Code Repositories

Answer technical questions about Ascend inference repositories with evidence-based, causally-grounded responses. DeepWiki is the primary knowledge source; embedded knowledge below is orientation context to help formulate better queries.

## Quick Diagnostic (for error/troubleshooting questions)

Before diving into deep research, complete this checklist — it resolves ~60% of issues and takes 2 minutes. Do not skip steps in favor of going straight to GitHub Issues.

1. **Version alignment** — vllm-ascend requires strict 1:1 match with upstream vLLM (e.g., v0.18.0 ↔ v0.18.0). Each release also pins exact CANN and torch_npu versions. Misalignment is the #1 root cause.
2. **Isolate graph mode** — add `--enforce-eager` to rule out ACLGraph/torchair as the cause.
3. **Check the error code** — ACL error codes point directly to the failure layer: `507015` (stream sync), `561000` (MoE dispatch), `503900` (KV transfer).
4. **Hardware platform** — 310P (Atlas 300I Duo) has a significantly smaller operator library than 910B. Features that work on 910B may simply not exist on 310P.
5. **Feature combination** — MTP + FULL_DECODE_ONLY, NZ layout + RL training, and W8A8 + MTP are known incompatible combinations.
6. **max_model_len** — silently too-large values cause OOM, garbled output, and shape errors. Always verify or suggest reducing it.
7. **Enable synchronous error reporting** — async execution hides the real error. Set these before reproducing:
   ```bash
   export ASCEND_LAUNCH_BLOCKING=1   # synchronous NPU execution — accurate stack traces
   export TASK_QUEUE_ENABLE=0        # disables async task queue
   export ASCEND_GLOBAL_LOG_LEVEL=0  # verbose CANN logs
   ```

Ask for `python collect_env.py` output (from the vllm-ascend repo root) when you need version details. Key fields to check: `vllm_ascend`, `vllm`, `torch_npu`, `CANN`, `driver`.

**Research strategy**: Complete the Quick Diagnostic checklist first, then use GitHub Issues search for version-specific bugs or recent fixes. Deep research should supplement standard checks, not replace them.

## 1. Repository Routing

Map user keywords to the correct DeepWiki `repoName` (exact `owner/repo` format required).

| User Keywords | DeepWiki `repoName` | Description |
|---|---|---|
| `vLLM` / `vllm` (no Ascend/NPU context) | `vllm-project/vllm` | Upstream vLLM engine |
| `vllm-ascend` / `vllm ascend` / `vLLM-Ascend` | `vllm-project/vllm-ascend` | Ascend NPU plugin for vLLM |
| `MindIE-LLM` / `mindie-llm` / `mindie llm` | `verylucky01/MindIE-LLM` | Ascend-native LLM inference engine |
| `SGLang` / `sglang` | `sgl-project/sglang` | High-performance serving framework with Ascend backend |
| `MindIE-SD` / `mindie-sd` | `verylucky01/MindIE-SD` | Diffusion model inference (SD, DiT, Wan2.x) |
| `MindIE-Motor` / `mindie-motor` (no `Py`) | `verylucky01/MindIE-Motor` | C++ serving framework |
| `MindIE-PyMotor` / `mindie-pymotor` (with `Py`) | `verylucky01/MindIE-PyMotor` | Python serving framework |
| `MindIE-Turbo` / `mindie-turbo` | `verylucky01/MindIE-Turbo` | Drop-in acceleration plugin for vLLM on Ascend |
| `msmodelslim` / `modelslim` / `MindStudio-ModelSlim` | `verylucky01/MindStudio-ModelSlim` | Model quantization toolkit for Ascend |

**Disambiguation** — ask only when genuinely ambiguous:
- `vllm` + NPU/昇腾/Ascend context → could be `vllm` or `vllm-ascend`; ask
- `MindIE` without suffix → ask which component (LLM / SD / Motor / PyMotor / Turbo)
- `Ascend` / `昇腾` / `NPU` alone without a project name → ask

For all other cases, proceed with the most likely repository.

## 2. Cross-Repo Query Strategies

### vllm-ascend: When to Also Query Upstream vLLM

`vllm-ascend` is a hardware plugin that overrides specific components (`NPUPlatform`, `NPUModelRunner`, `AscendAttentionBackend`, `ACLGraph`) while inheriting most logic from upstream vLLM.

**Query both repos (upstream first, then vllm-ascend) when:**
- Model support, scheduling, sampling, or API behavior (mostly inherited from upstream)
- How vllm-ascend modifies upstream behavior (attention backend, worker, model runner)
- Architecture questions crossing the plugin boundary

**Query vllm-ascend alone when:**
- Ascend-specific config: `AscendConfig`, graph mode, NPU env vars
- Installation, Docker images, version compatibility
- ACLGraph, torchair, HCCL setup, Ascend-specific bugs

When both repos are consulted, attribute each statement explicitly: "In upstream vLLM, ..." vs. "In vllm-ascend, ...".

### MindIE-Turbo: Layered Queries

MindIE-Turbo modifies vLLM at runtime via a `Patcher` registry. Questions about how Turbo alters vLLM behavior may require querying both `vllm-project/vllm` and `verylucky01/MindIE-Turbo`.

### MindIE Motor/PyMotor: Know What Sits Below

- Motor wraps MindIE-LLM via C++-to-C++ coupling
- PyMotor integrates vLLM-Ascend or SGLang as backends

Serving behavior questions → Motor/PyMotor. Inference behavior questions → the underlying engine.

### Cross-Repo Comparisons

Query both repos independently and in parallel, then integrate into a structured comparison.

## 3. Architecture Context

Orientation knowledge to help formulate DeepWiki queries. Always verify against DeepWiki before citing in responses — do not cite this section as a source.

### vllm-ascend Key Components

- **NPUPlatform**: Device management, registered via `setup.py` entry point `ascend = vllm_ascend:register`
- **NPUModelRunner**: Extends GPUModelRunner with NPU forward contexts, rotary embedding, weight prefetch (V1/V2 variants)
- **AscendAttentionBackend / AscendMLABackend**: Attention on NPU; supports host-side parameter updates during graph replay via `graph_task_update`
- **ACLGraph / ACLGraphWrapper**: Captures and replays NPU computation graphs via `torch.npu.NPUGraph`; keyed by `BatchDescriptor` (not just batch size)
- **torchair / Npugraph_ex**: Compile-time FX graph optimization; default in `FULL` and `FULL_DECODE_ONLY` graph modes
- **AscendConfig**: Central configuration with nested configs for compilation, graph mode, parallelism, and env vars

### MindIE-LLM Architecture

- C++ scheduling core (LLM Manager) for batch management and KV cache block allocation
- Python text generator layer for model orchestration
- ATB/AclGraph backends for NPU kernel execution
- OpenAI-compatible RESTful API
- Supports continuous batching (FCFS/PDDS), PagedAttention with prefix caching, TP/DP/EP/CP/PD disaggregation

### ModelSlim Quantization Pipeline

- Exports quantized weights in **AscendV1 format** (`quant_model_description.json` + `quant_model_weight_<quant_type>.safetensors`)
- vllm-ascend loads via `--quantization ascend`; MindIE-LLM loads directly
- Supported methods: SmoothQuant, AWQ, GPTQ, AutoRound, LinearQuant, FA3 (per-head INT8 for MLA), KVCache Quant, PDMIX, W4A4 (LAOS)
- Conversion tool `ms_to_vllm.py` for AWQ/GPTQ → vLLM-native format

### MindIE-Turbo Optimization Levels

- L0: Basic (debug-friendly) / L1: Stable production / L2: Aggressive (default) / L3: Experimental
- Set via `VLLM_OPTIMIZATION_LEVEL` environment variable

### Repository Relationships

```
Users / API Clients
  │
  ├── MindIE-Motor (C++)  or  MindIE-PyMotor (Python)   ← serving/orchestration
  │     ├── MindIE-LLM                                  ← Ascend-native LLM engine
  │     └── vLLM-Ascend + optional MindIE-Turbo         ← patched vLLM path
  ├── SGLang                                            ← serving framework with Ascend backend
  └── MindIE-SD                                         ← diffusion model inference (independent)

ModelSlim → quantize → any inference engine above
```

## 4. Query Formulation

Always construct DeepWiki queries in **English** using precise technical terminology, regardless of the user's input language.

| Problem Type | Query Approach |
|---|---|
| Deployment / Installation | "Deployment guide and installation prerequisites for [repo]" |
| Error / Stack trace | Extract the key error class/message. For Ascend: look for `RuntimeError` from `torch_npu`, CANN error codes (`Exxxxx` or numeric like `507015`), HCCL errors, `AscendError`. Query: "[ErrorType] causes and solutions" |
| Architecture not recognized | "Supported model architectures" + "version compatibility with transformers" |
| Performance | "Performance optimization: [specific aspect — batch size, KV cache, graph mode, prefix cache]" |
| Configuration | "How to configure [parameter] for [use case]" |
| Model support | "Supported model architectures and compatibility matrix" |
| Source code | "Implementation of [component] and interaction with [related component]" |
| Version compatibility | "Version compatibility requirements between [component A] and [component B]" |
| MTP / Speculative decoding | "Multi-token prediction and speculative decoding configuration" |
| PD disaggregation | "Prefill-decode disaggregation deployment and KV transfer configuration" |
| Tool call / function calling | "[Model name] tool calling support and configuration" |
| Multimodal / VL | "[Model name] vision-language inference configuration and known issues" |
| LoRA | "LoRA and multi-LoRA serving configuration" |
| Graph mode | "ACLGraph and FULL_DECODE_ONLY graph capture mode configuration" |

If a single query yields insufficient information, explore from multiple perspectives. When unsure which section covers the topic, start with `mcp__deepwiki__read_wiki_structure`.

## 5. DeepWiki Tool Selection

| Scenario | Tool |
|---|---|
| Specific question with clear direction | `ask_question` |
| Unsure which section covers the topic | `read_wiki_structure` → then `ask_question` |
| Need comprehensive coverage after multiple insufficient `ask_question` calls | `read_wiki_contents` (use sparingly) |
| Multiple independent repos | Parallel `ask_question` calls, or pass `repoName` as array |

Reuse results from earlier in the conversation when the same topic was already queried.

**When DeepWiki yields no results**: rephrase or expand the query. If still no results, escalate to supplementary sources — don't just note the gap and proceed with embedded knowledge alone.

**Supplementary sources** — use proactively for version-specific bugs, recent fixes, and error codes, not just as a fallback:
- **GitHub Issues** (for `vllm-project/*` repos): `gh issue list --repo <owner/repo> --search "<keywords>" --state all`. Use this proactively for: specific error codes (e.g., "507015"), feature combination bugs (e.g., "MTP FULL_DECODE_ONLY"), and hardware-specific issues (e.g., "310P TransData"). Prompt user for a GitHub Token if needed.
- **GitCode Issues** (for `verylucky01/*` repos — replace `verylucky01` with `Ascend` in the API path): `curl --location 'https://api.gitcode.com/api/v5/search/issues?q=<keywords>&repo=<Ascend/repo>' --header 'Authorization: Bearer <Your-Token>'`. Prompt user for a GitCode Token.
- **Local code**: If working within a cloned repo, reading source files directly is more authoritative than DeepWiki.

When citing information, always attribute to the actual source (DeepWiki, GitHub Issues, source file path) — never cite this skill document as a source in your response.

## 6. Information Gathering for Troubleshooting

For troubleshooting, deployment, configuration, or performance questions, request the most relevant subset of:

- **Hardware**: Ascend chip model and card count
  - Atlas 800I A2 / Atlas 800T A2 = **910B** (910B2/910B3/910B4, most common)
  - Atlas 800I A3 / A3 Ultra Node = **910C**
  - Atlas 300I Duo = **310P**
- **Software versions**: CANN, torch, torch_npu, transformers, vLLM/MindIE version, vllm-ascend version, Triton-Ascend
- **OS**: Linux distribution and kernel version (GLIBC version matters for container compatibility)
- **Docker image tag** (if applicable)
- **Error context**: Full error messages, log snippets, stack traces
- **Startup command**: Full `vllm serve` or equivalent command with all flags
- **Deployment topology**: Single-node vs. multi-node, TP/DP/EP/PP degrees, PD disaggregation setup

Suggest `python collect_env.py` (from vllm-ascend repo root) to collect comprehensive environment info. When interpreting the output, focus on: `vllm_ascend` version (must match `vllm`), `cann_version` (must match `torch_npu` compatibility matrix), `npu_driver_version` (must match CANN), and `torch_npu` version.

**Version alignment is the #1 root cause** — roughly one-third of all reported issues. vllm-ascend maintains strict 1:1 version correspondence with upstream vLLM, and each release specifies exact compatible CANN and torch_npu versions.

## 7. Common Problem Patterns

Ranked by real-world frequency based on 520+ closed vllm-ascend issues. Check the most frequent patterns first. Verify specific details against DeepWiki or GitHub Issues before citing.

| Symptom | Likely Causes | Investigation Direction |
|---|---|---|
| **Model startup / launch failure** | Version mismatch (vLLM ↔ vllm-ascend strict 1:1), unsupported architecture, incorrect quantization format, missing compiled extension (`No module named 'vllm_ascend.*'`) | Check vllm-ascend release notes for version alignment; verify model architecture support; check for `ModuleNotFoundError` |
| **"does not recognize this architecture"** | `transformers` version too old for the model type (e.g., Qwen3-VL, DeepSeek-V4, GLM-5) | Upgrade transformers: `pip install --upgrade transformers`; verify model is supported in the vllm-ascend version |
| **Version / compatibility mismatch** | vLLM ↔ vllm-ascend version mismatch; CANN incompatible with torch_npu; driver version not supported by CANN; numpy 2.x (`np.float_` removed); `torch not compiled with CUDA enabled` (wrong torch build) | Check vllm-ascend release notes for the exact CANN + torch_npu + vLLM version matrix; verify `torch_npu` is the NPU build, not CUDA |
| **ACL / CANN runtime errors** | CANN version incompatible with torch_npu or driver; operator not found in op store; stream sync failure | Common error codes: `507015` (ACL stream sync), `561000` (MoE dispatch), `503900` (KV transfer). Verify CANN-torch_npu-driver compatibility matrix; check `aclnn*` operator availability for the target chip |
| **FULL_DECODE_ONLY / graph mode failure** | CANN version incompatible with torchair; model not supported in graph mode; `VLLM_ASCEND_ENABLE_NZ=1` interaction with MTP or other features; disk space exhaustion during graph capture on A3 | Disable graph mode first to isolate (`--enforce-eager`); check CANN version compatibility with torchair; verify feature combination support |
| **MTP / Speculative decoding errors** | MTP + FULL_DECODE_ONLY combination not supported; Eagle3 incompatibility with certain model configs; `NPUModelRunner has no attribute attn_metadata_builder`; acceptance rate drops with quantized models | Check vllm-ascend-specific speculative config (may differ from upstream); verify feature combination support; try disabling MTP first to isolate |
| **OOM (Out of Memory)** | `max_model_len` too large; `block_size` misconfigured; TP degree insufficient; graph capture memory competing with KV cache; long context causing memory explosion | Reduce `max_model_len`; increase TP; check `gpu_memory_utilization`; see `VLLM_ASCEND_MEMORY_PROFILER_ESTIMATE_NPUGRAPHS` |
| **PD disaggregation / Mooncake failures** | Port conflict when P and D on same machine; `Transfer slice failed with status: 503900`; KV pool blocking requests; DP size mismatch in KV transfer config; Mooncake broken in specific image versions | Verify P/D port configuration; check DP/TP settings match between prefill and decode instances; verify Mooncake image version compatibility |
| **Multi-node / distributed failures** | Ray startup method mismatch; `ASCEND_RT_VISIBLE_DEVICES` misconfigured; HCCL errors (`HCCL function error`, `HCCL_BUFFSIZE` not verified); `stateless_init_process_group is invalid on NPUs`; MoE EP init repeated initialization | Verify Ray setup matches deployment topology; check NPU device visibility; verify HCCL environment variables; check `cannot set moe all to all group` for EP init ordering |
| **Multimodal / VL model issues** | Shape errors during `_prepare_inputs` under high concurrency; video inference not supported in certain versions; `merge_by_field_config` not supported; garbled output ("!!!!") in VL models | Check model-specific VL support status in the vllm-ascend version; reduce concurrency to isolate shape errors; verify `max_model_len` is not silently too large |
| **Quantization issues** | Wrong export format (must be AscendV1); incompatible quantization method for target engine; W8A8 + MTP combination bugs; `--quantization ascend` option not available in older versions | For vllm-ascend: ensure AscendV1 format, use `--quantization ascend`; verify quantization method is supported for the target model and chip |
| **LoRA / multi-LoRA issues** | `ValueError: expected target modules` mismatch; slow inference with LoRA on 310P; multi-LoRA crashes on certain model+version combinations | Check LoRA target module names match the model architecture; verify LoRA support status for the specific model in the vllm-ascend version |
| **310P / Atlas 300I hardware gaps** | Triton-Ascend not supporting 310P; `TransData op not found in op store`; features working on 910B but broken/unsupported on 310P; `kernel_macros.h` errors; `VLLM_ASCEND_ENABLE_NZ=1` not supported on 310P | Try `VLLM_ASCEND_ENABLE_NZ=0` first (works on some versions); use 310P-specific Docker image tags; add `--enforce-eager --dtype float16`; note that 310P support was dropped after v0.10.0rc1 — check GitHub Issues for the exact last supported version |
| **Garbled / empty / repetitive output** | `max_model_len` silently too large; quantization precision loss; `fp16` producing "!!!!" output (known pattern); hardware-specific numerical issues; async-scheduling flag causing precision issues | Reduce `max_model_len`; try FP16/BF16 baseline before quantized; disable `async-scheduling`; check for model-specific patches in vllm-ascend |
| **Slow inference / performance regression** | Graph mode not enabled; suboptimal batch config; CPU binding overlap between NPU instances; prefix cache causing regression under concurrent load; MindIE-Turbo not taking effect | Enable graph mode (`FULL` or `FULL_DECODE_ONLY`); tune batch parameters; check CPU affinity; verify MindIE-Turbo is loaded (`VLLM_OPTIMIZATION_LEVEL`) |
| **Installation / build failures** | `cmake>=3.26.1` not found; `torch_npu` version pinning conflicts; `GLIBC_2.3x not found` (container/host OS mismatch); `undefined symbol` in `libop_plugin_atb.so` or `mindie_turbo`; `No module named 'numba'` | Check build prerequisites; verify GLIBC version compatibility between container and host; use official Docker images when possible |
| **VERL / RL training integration** | Rollout parameter update failures with `NZ+matmul_allreduce`; `EngineCore process coredump` under RL workloads; `assert self.cpu_group is not None` | Check `VLLM_ASCEND_ENABLE_NZ` interaction with RL frameworks; verify veRL version compatibility with vllm-ascend version |

## 8. Response Standards

- **Conclusion first**: Begin with the direct answer, then supporting details.
- **Traceability**: Cite the actual source — DeepWiki query results, GitHub Issues, or source file paths. Do not cite this skill document.
- **Attribution**: For vllm-ascend topics, explicitly distinguish upstream vLLM information from plugin-specific content.
- **Accuracy**: Technical details must align with verified sources. Acknowledge gaps rather than fabricating details. Mark unverified embedded knowledge: "(unverified — please confirm via official docs or source code)".
- **Completeness**: Proactively address prerequisites, background context, and common pitfalls.
- **Scope**: This skill applies only to these nine repositories. For out-of-scope questions, state this clearly.
