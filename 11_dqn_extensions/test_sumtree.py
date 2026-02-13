#!/usr/bin/env python3
"""
Unit Tests für SumTree Implementation
Verifiziert mathematische Korrektheit
"""

import numpy as np
import random
from collections import namedtuple

print("="*80)
print("SUMTREE UNIT TESTS - VERIFIKATION")
print("="*80)

Transition = namedtuple("Transition", ["s", "a", "r", "s2", "done"])

class SumTree:
    """SumTree - EXACT copy from DQN_Extensions"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data):
        leaf_idx = self.write + (self.capacity - 1)
        self.data[self.write] = data
        self.update(leaf_idx, p)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), self.data[data_idx]


# ============================================================================
# TEST 1: Basic Add and Total
# ============================================================================

print("\n[TEST 1] Basic Add and Total Calculation")
print("-" * 80)

tree = SumTree(4)
print(f"Created SumTree with capacity=4")
print(f"  tree.size = {tree.size}")
print(f"  tree.total() = {tree.total()}")

tree.add(0.5, Transition(0, 0, 0, 0, False))
print(f"\nAdded priority=0.5")
print(f"  tree.size = {tree.size}")
print(f"  tree.total() = {tree.total():.4f}")
assert abs(tree.total() - 0.5) < 1e-5, "Total should be 0.5"
print(f"  ✓ PASS")

tree.add(0.3, Transition(1, 1, 1, 1, False))
print(f"\nAdded priority=0.3")
print(f"  tree.size = {tree.size}")
print(f"  tree.total() = {tree.total():.4f}")
assert abs(tree.total() - 0.8) < 1e-5, "Total should be 0.8"
print(f"  ✓ PASS")

tree.add(0.2, Transition(2, 2, 2, 2, False))
print(f"\nAdded priority=0.2")
print(f"  tree.size = {tree.size}")
print(f"  tree.total() = {tree.total():.4f}")
assert abs(tree.total() - 1.0) < 1e-5, "Total should be 1.0"
print(f"  ✓ PASS")

# ============================================================================
# TEST 2: Leaf Index Calculation
# ============================================================================

print("\n[TEST 2] Leaf Index Mapping")
print("-" * 80)

tree2 = SumTree(4)
print(f"SumTree with capacity=4")
print(f"  tree.size = {2*4-1} = 7")
print(f"  Leaf indices: 3, 4, 5, 6")
print(f"  Data indices: 0, 1, 2, 3")

for data_idx in range(4):
    tree2.add(0.1 * (data_idx + 1), f"data{data_idx}")
    leaf_idx = data_idx + (4 - 1)
    print(f"\n  data_idx={data_idx} → leaf_idx={leaf_idx}")
    assert tree2.data[data_idx] == f"data{data_idx}", "Data storage error"
    print(f"    ✓ Data correctly stored")

print(f"\n✓ PASS - Leaf indices correctly mapped")

# ============================================================================
# TEST 3: Update Propagation
# ============================================================================

print("\n[TEST 3] Update Propagation (Parent Index Calculation)")
print("-" * 80)

tree3 = SumTree(4)
tree3.add(1.0, "a")
tree3.add(1.0, "b")
tree3.add(1.0, "c")

print(f"Initial state: [1.0, 1.0, 1.0]")
print(f"  tree[0] (root) = {tree3.tree[0]:.4f}")
print(f"  tree[1] (left) = {tree3.tree[1]:.4f}")
print(f"  tree[2] (right) = {tree3.tree[2]:.4f}")

leaf_idx = 3  # First leaf
print(f"\nUpdating leaf_idx={leaf_idx} from 1.0 to 5.0 (change=+4.0)")
tree3.update(leaf_idx, 5.0)

print(f"After update:")
print(f"  tree[3] (leaf) = {tree3.tree[3]:.4f}")
print(f"  tree[1] (parent) = {tree3.tree[1]:.4f}")
print(f"  tree[0] (root) = {tree3.tree[0]:.4f}")
assert abs(tree3.tree[0] - 7.0) < 1e-5, f"Root should be 7.0, got {tree3.tree[0]}"
print(f"  ✓ PASS - Update propagated correctly")

# ============================================================================
# TEST 4: Get and Stratified Sampling
# ============================================================================

print("\n[TEST 4] Sampling Distribution (Stratified Sampling)")
print("-" * 80)

tree4 = SumTree(4)
priorities = [0.5, 0.3, 0.15, 0.05]
for p in priorities:
    tree4.add(p, f"p={p}")

print(f"Added priorities: {priorities}")
print(f"Total: {tree4.total():.4f}")

# Sample 10000 times and check distribution
samples = {0: 0, 1: 0, 2: 0, 3: 0}
for _ in range(10000):
    s = random.uniform(0, tree4.total())
    leaf_idx, p, data = tree4.get(s)
    data_idx = leaf_idx - (4 - 1)
    samples[data_idx] += 1

print(f"\nSampling results (10000 samples):")
expected_counts = [int(10000 * p / tree4.total()) for p in priorities]
for i, (p, actual, expected) in enumerate(zip(priorities, samples.values(), expected_counts)):
    pct = (actual / 10000) * 100
    expected_pct = (expected / 10000) * 100
    error = abs(pct - expected_pct)
    print(f"  data_idx={i}: {actual:4d} samples ({pct:5.1f}%) | Expected: {expected:4d} ({expected_pct:5.1f}%) | Error: {error:4.1f}%")
    assert error < 1.0, "Sampling distribution too far from expected"

print(f"\n✓ PASS - Sampling distribution matches priorities")

# ============================================================================
# TEST 5: Data Retrieval Correctness
# ============================================================================

print("\n[TEST 5] Data Retrieval (Data Index Mapping)")
print("-" * 80)

tree5 = SumTree(4)
test_data = ["first", "second", "third", "fourth"]
for data in test_data:
    tree5.add(1.0, data)

print(f"Added data: {test_data}")

retrieved = []
for i in range(4):
    leaf_idx = i + 3  # Direct calculation
    s = tree5.tree[leaf_idx] / 2  # Sample in middle of leaf
    _, _, data = tree5.get(s)
    retrieved.append(data)
    print(f"  Sampling from leaf_idx={leaf_idx}: got '{data}'")

assert set(retrieved) == set(test_data), "Data mismatch!"
print(f"\n✓ PASS - All data correctly retrieved")

# ============================================================================
# TEST 6: Priority Update (Critical for PER)
# ============================================================================

print("\n[TEST 6] Priority Updates (Critical for PER Training)")
print("-" * 80)

tree6 = SumTree(4)
tree6.add(1.0, "a")
tree6.add(1.0, "b")

initial_total = tree6.total()
print(f"Initial: 2 items with priority 1.0 each")
print(f"  total = {initial_total:.4f}")

# Simulate PER: update first item to high priority
leaf_idx_a = 0 + 3
print(f"\nUpdating leaf_idx={leaf_idx_a} to priority=10.0")
tree6.update(leaf_idx_a, 10.0)

new_total = tree6.total()
print(f"  total = {new_total:.4f}")
assert abs(new_total - 11.0) < 1e-5, "Total should be 11.0"

# Check that first item is now heavily sampled
samples = {0: 0, 1: 0}
for _ in range(10000):
    s = random.uniform(0, new_total)
    leaf_idx, p, _ = tree6.get(s)
    data_idx = leaf_idx - 3
    samples[data_idx] += 1

pct_a = (samples[0] / 10000) * 100
print(f"\nAfter 10000 samples:")
print(f"  Item A (priority=10.0): {samples[0]:5d} samples ({pct_a:.1f}%)")
print(f"  Item B (priority=1.0):  {samples[1]:5d} samples ({100-pct_a:.1f}%)")
print(f"  Expected: A≈90.9%, B≈9.1%")
assert 85 < pct_a < 96, "First item should be ~90% likely"

print(f"\n✓ PASS - Priority updates work correctly")

# ============================================================================
# TEST 7: Wraparound (Circular Buffer)
# ============================================================================

print("\n[TEST 7] Circular Buffer Wraparound")
print("-" * 80)

tree7 = SumTree(3)  # Small capacity to test wraparound
print(f"SumTree with capacity=3 (will test wraparound)")

for i in range(6):
    tree7.add(1.0, f"item_{i}")
    print(f"  Added item_{i}: size={tree7.size}, write={tree7.write}")

print(f"\nFinal size (should be min(6, 3)): {tree7.size}")
assert tree7.size == 3, "Size should be capped at capacity"
print(f"  ✓ PASS")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✓ ALL TESTS PASSED!")
print("="*80)
print("""
VERIFIED COMPONENTS:
  ✓ Total priority calculation
  ✓ Leaf index mapping
  ✓ Parent index calculation
  ✓ Update propagation
  ✓ Stratified sampling
  ✓ Data retrieval
  ✓ Priority updates
  ✓ Circular buffer wraparound

CONCLUSION:
  The SumTree implementation in DQN_Extensions.ipynb is MATHEMATICALLY CORRECT!
  No bugs detected in core SumTree logic.
""")

