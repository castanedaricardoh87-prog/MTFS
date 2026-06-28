import torch

def compute_dipolar_coherence(d1, d2):
    """Simple coherence between two dipolar signatures."""
    d1_n = torch.nn.functional.normalize(d1, p=2, dim=-1)
    d2_n = torch.nn.functional.normalize(d2, p=2, dim=-1)
    return (d1_n * d2_n).sum(dim=-1)

# Marvo Mini Demo
print("=== MTFS Marvo Mini ===")
print("Basic dipolar coherence + equilibrium demo\n")

# Example states (S, A, R components)
state1 = torch.tensor([1.0, 0.5, 0.2])
state2 = torch.tensor([0.9, 0.6, 0.3])

# Simple dipolar projection (D+ , D-)
d1 = torch.tensor([state1[0] - state1[1], state1[1] - state1[2]])
d2 = torch.tensor([state2[0] - state2[1], state2[1] - state2[2]])

coh = compute_dipolar_coherence(d1, d2)
print(f"Dipolar Signature 1: {d1}")
print(f"Dipolar Signature 2: {d2}")
print(f"Coherence: {coh.item():.4f} (closer to 1 = better alignment)")

# Basic equilibrium example
def simple_equilibrium(s, a, r, q=0.1, s0=1.0):
    m = s + a + r
    ds = -q * (s - s0)
    return m, ds

m, ds = simple_equilibrium(0.8, 0.3, 0.2)
print(f"\nEquilibrium M(t): {m:.4f}, Damping: {ds:.4f}")
