"""Verify complete removal of BGE model integration and correct initialization of ONNX and Baseline rerankers."""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.rag.reranker import get_reranker, ONNXReranker, BaselineReranker, _RERANKER_CACHE
from app.config import settings

print("=" * 60)
print("TARK Studio — Post-BGE Removal Verification")
print("=" * 60)

# 1. Baseline Reranker
print("\n1. Testing Baseline reranker...")
_RERANKER_CACHE.clear()
base = get_reranker("baseline")
assert isinstance(base, BaselineReranker)
print("   [PASS] BaselineReranker initialized correctly.")

# 2. ONNX Reranker
print("\n2. Testing ONNX reranker initialization...")
onnx = get_reranker("onnx")
assert isinstance(onnx, ONNXReranker)
print(f"   [PASS] ONNXReranker initialized. Model load time: {onnx.model_load_time_sec}s")

# 3. Refusal of removed BGE type
print("\n3. Testing refusal of removed BGE type...")
try:
    get_reranker("bge")
    assert False, "Should have raised ValueError"
except ValueError as e:
    assert "Supported: 'baseline', 'onnx'" in str(e)
    print(f"   [PASS] Correctly rejected 'bge': {e}")

# 4. ONNX Reranking scoring
print("\n4. Testing ONNX scoring with sample chunks...")
class MockChunk:
    def __init__(self, content):
        self.content = content

chunks = [
    MockChunk("Python is an interpreted, high-level, general-purpose programming language."),
    MockChunk("Bananas and apples are common fruits found in grocery stores."),
]
results = onnx.rerank("What programming language is interpreted?", chunks, top_k=2)
assert "Python" in results[0].content
assert results[0].rerank_score > results[1].rerank_score
print(f"   [PASS] Top passage: '{results[0].content[:40]}...' (score={results[0].rerank_score:.2f})")
print(f"   [PASS] Second passage: '{results[1].content[:40]}...' (score={results[1].rerank_score:.2f})")

# 5. Check configuration
print("\n5. Checking application settings...")
assert not hasattr(settings, "reranker_model_bge"), "settings still has reranker_model_bge!"
print(f"   [PASS] settings.reranker_type = '{settings.reranker_type}'")
print(f"   [PASS] settings.reranker_model_onnx = '{settings.reranker_model_onnx}'")
print("   [PASS] No reranker_model_bge attribute found in settings.")

print("\n" + "=" * 60)
print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
print("=" * 60)
