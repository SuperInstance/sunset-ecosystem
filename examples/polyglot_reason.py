"""examples/polyglot_reason.py — Polyglot reasoning example.

Demonstrates Rust/C++/Python/Mercury backends for tile similarity.

Usage:
    python examples/polyglot_reason.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from reasoning.python_bridge import PolyglotReasoner

def benchmark_backend(reasoner, name, iterations=1000):
    """Benchmark a backend."""
    query = [1.0] + [0.0] * (reasoner.dim - 1)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        reasoner.find_similar(query, top_k=5)
    dt = time.perf_counter() - t0
    
    print(f"  {name}: {dt*1000:.2f}ms ({iterations/dt:.0f} ops/sec)")

def main():
    print("🌐 Polyglot Reasoner Demo")
    print("=" * 40)
    
    dim = 256
    num_tiles = 1000
    
    print(f"\n📊 Setup: {num_tiles} tiles, dim={dim}")
    
    # Create tiles
    import numpy as np
    tiles = []
    for i in range(num_tiles):
        emb = np.random.randn(dim).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        tiles.append((i, emb.tolist()))
    
    # Test Python backend
    print("\n🐍 Python backend:")
    py_reasoner = PolyglotReasoner(dim=dim, backend="python")
    for tid, emb in tiles:
        py_reasoner.add_tile(tid, emb)
    benchmark_backend(py_reasoner, "Python", 100)
    
    # Test Rust backend (if available)
    if py_reasoner.get_stats()["rust_available"]:
        print("\n⚙️  Rust backend:")
        rust_reasoner = PolyglotReasoner(dim=dim, backend="rust")
        for tid, emb in tiles:
            rust_reasoner.add_tile(tid, emb)
        benchmark_backend(rust_reasoner, "Rust", 100)
    
    # Test C++ backend (if available)
    if py_reasoner.get_stats()["cpp_available"]:
        print("\n🏎️  C++ backend:")
        cpp_reasoner = PolyglotReasoner(dim=dim, backend="cpp")
        for tid, emb in tiles:
            cpp_reasoner.add_tile(tid, emb)
        benchmark_backend(cpp_reasoner, "C++", 100)
    
    # Test query
    print("\n🔍 Query: find similar to tile 0")
    query = tiles[0][1]
    results = py_reasoner.find_similar(query, top_k=5)
    print(f"  Top 5 matches: {results}")
    
    # Test Mercury verification (if available)
    print("\n🔮 Mercury verification:")
    result = py_reasoner.verify_with_mercury(query, 0)
    print(f"  Verified: {result['verified']}")
    if not result['verified']:
        print(f"  Reason: {result['reason']}")
    
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()
