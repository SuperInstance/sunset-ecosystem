# Plato Construct Expansion — Architecture v2

## Overview

Plato Construct v2 transforms the Python-only tile engine into a **polyglot, voice-enabled, JEPA-powered room system** that supports both local model inference and API fallback, with real-time voice interaction via soniqo.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Voice Layer (soniqo)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │   ASR   │  │   TTS   │  │   VAD   │                 │
│  │ speech→ │  │ text→   │  │ voice   │                 │
│  │  text   │  │ speech  │  │ detect  │                 │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
│       └─────────────┴─────────────┘                      │
│                Voice Tiles                               │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                    JEPA Room Layer                       │
│  ┌─────────────────┐    ┌─────────────────┐             │
│  │  Local Encoder  │    │  API Fallback   │             │
│  │  (on-device)    │◄──►│  (cloud)        │             │
│  │                 │    │                 │             │
│  │  JEPA Predictor │    │  Gemini/Claude  │             │
│  │  next-tile      │    │  etc.           │             │
│  └─────────────────┘    └─────────────────┘             │
│  Confidence gating: JEPA > threshold → local, else API   │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                Polyglot Reasoning Layer                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │
│  │  Rust   │  │  C++    │  │ Python  │  │ Mercury │   │
│  │ SIMD    │  │ OpenMP  │  │ NumPy   │  │ Formal  │   │
│  │ fast    │  │ GPU     │  │ fallback│  │ verify  │   │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘   │
│  Auto-selection: Rust → C++ → Python → Mercury            │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                    Core Plato Engine                     │
│  Tiles · Rooms · Prompts · Bot · LLM · Surface · Log  │
└─────────────────────────────────────────────────────────┘
```

## Module Inventory

| Module | Language | Tests | Purpose |
|--------|----------|-------|---------|
| `voice/soniqo_bridge.py` | Python | 16 | Voice integration with ASR/TTS/VAD |
| `jepa/jepa_room.py` | Python | 16 | Local JEPA inference + API fallback |
| `reasoning/rust/src/lib.rs` | Rust | 4 | SIMD tile similarity |
| `reasoning/cpp/reasoner.cpp` | C++ | 4 | OpenMP batch similarity |
| `reasoning/mercury/reasoner.m` | Mercury | 5 | Formal verification |
| `reasoning/python_bridge.py` | Python | 11 | Unified polyglot interface |

## Key Design Decisions

### 1. Hybrid Local/API Architecture
JEPA rooms use local models for fast, private inference. When confidence is below threshold, they seamlessly fall back to API. This provides both speed and accuracy.

### 2. Voice as First-Class Tiles
Every voice interaction is captured as a VoiceTile with transcript, audio hash, and metadata. Voice is not an add-on; it's a core tile type.

### 3. Polyglot Reasoning
Not all code needs to be Python. Performance-critical paths are in Rust (SIMD), GPU-heavy operations in C++ (OpenMP), and verification in Mercury (formal proofs). The Python bridge auto-selects the best available backend.

### 4. Graceful Degradation
Every component has a mock fallback:
- soniqo → mock ASR/TTS/VAD
- JEPA → mock encoder/predictor  
- Rust/C++ → Python NumPy fallback
- Mercury → skip verification

## Confidence Gating

| Confidence | Action |
|------------|--------|
| > 0.9 | JEPA local inference |
| 0.7 - 0.9 | JEPA + API ensemble |
| < 0.7 | API fallback only |
| < 0.5 | Human escalation |

## FM Testing Gates

1. **soniqo SDK**: Install speech-swift or speech-android for real voice tests
2. **JEPA models**: Load encoder/predictor checkpoints from PyTorch
3. **Rust reasoner**: `cargo build --release` in `reasoning/rust/`
4. **C++ reasoner**: `g++ -O3 -fopenmp -shared -fPIC -o libplato_cpp.so reasoner.cpp`
5. **Mercury reasoner**: `mmc --make reasoner` in `reasoning/mercury/`

## Success Metrics

- Voice latency: < 500ms end-to-end
- JEPA accuracy: > 85% on tile sequences
- Rust speedup: 10x vs Python on benchmark
- All tests: 100% green (mock + real)

## Next Steps

1. FLUX voice gating: constraints on voice generation
2. Mercury voice spec: formal protocol for voice tiles
3. JEPA model training: fine-tune on fleet tile corpus
4. GPU acceleration: CUDA kernels for C++ reasoner
