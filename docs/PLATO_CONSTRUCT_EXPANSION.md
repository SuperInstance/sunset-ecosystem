# PLATO Construct Expansion — Strategic Plan

## Vision
Transform plato-construct from a Python-only tile engine into a polyglot, voice-enabled, JEPA-powered room system that can run local models (JEPA) in some rooms and API models in others, with real-time voice interaction via soniqo.

## Phase 1: Soniqo Integration Bridge (P0)
Real-time voice interaction with PLATO rooms.

- `voice/soniqo_bridge.py` — Python bridge to soniqo SDKs
- `voice/room_voice.py` — Voice-enabled room wrapper
- `voice/jepa_voice.py` — JEPA audio encoding for voice tiles
- Tests: 15+ covering ASR, TTS, VAD integration

## Phase 2: JEPA Room System (P0)
JEPA (Joint Embedding Predictive Architecture) rooms for local model inference.

- `jepa/jepa_room.py` — JEPA-powered room with local encoder/predictor
- `jepa/jepa_tile_encoder.py` — Encode tiles to JEPA latent space
- `jepa/jepa_predictor.py` — Predict next tile from context
- `jepa/api_fallback.py` — Fallback to API when local model insufficient
- Tests: 20+ covering encoding, prediction, fallback

## Phase 3: Polyglot Reasoning Layer (P1)
Low-level reasoning in multiple languages.

- `reasoning/rust_reasoner.rs` — Rust reasoner for performance-critical paths
- `reasoning/cpp_reasoner.cpp` — C++ reasoner for GPU-accelerated inference
- `reasoning/mercury_reasoner.m` — Mercury reasoner for formal verification
- `reasoning/python_bridge.py` — Unified bridge to all reasoners
- Tests: 25+ per language

## Phase 4: Documentation & Examples (P1)
- `docs/ARCHITECTURE-v2.md` — Complete architecture documentation
- `docs/VOICE_INTEGRATION.md` — Soniqo integration guide
- `docs/JEPA_ROOMS.md` — JEPA room setup and usage
- `docs/POLYGLOT_REASONING.md` — Multi-language reasoning guide
- `examples/voice_room.py` — Voice-enabled room example
- `examples/jepa_chat.py` — JEPA chat room example
- `examples/polyglot_reason.py` — Multi-language reasoning example

## Phase 5: FLUX & Mercury Integration (P2)
- `flux/flux_voice_gate.py` — FLUX constraints on voice generation
- `mercury/mercury_voice_spec.m` — Mercury specification for voice protocols
- `mercury/mercury_jepa_spec.m` — Mercury specification for JEPA semantics

## Build Order
1. Soniqo bridge (voice → tiles)
2. JEPA room core (local inference)
3. API fallback (hybrid local/API)
4. Rust reasoner (performance)
5. C++ reasoner (GPU)
6. Mercury reasoner (verification)
7. Documentation
8. FLUX/Mercury integration

## Deliverables
- 12 new modules
- 100+ tests
- 4 documentation files
- 3 example scripts
- 1 integration spec

## FM Gates
- Soniqo SDKs must be installed for real voice tests
- JEPA models need torch/MLX for local inference
- Rust/C++ reasoners need cargo/gcc for compilation
- Mercury reasoner needs mmc for verification

## Success Metrics
- Voice tiles submitted in <500ms
- JEPA prediction accuracy >85% on tile sequences
- Rust reasoner 10x faster than Python on benchmark
- All tests green, documentation complete
