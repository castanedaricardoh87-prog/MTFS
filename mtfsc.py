import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniMTFS(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.signature = nn.Linear(dim, 2)

    def forward(self, state):
        return self.signature(state)

class FractalNode(nn.Module):
    def __init__(self, level=1, max_level=4):
        super().__init__()
        self.level = level
        self.is_leaf = level >= max_level
        if self.is_leaf:
            self.mini = MiniMTFS()
        else:
            self.sub_nodes = nn.ModuleList([FractalNode(level+1, max_level) for _ in range(3)])

    def forward(self, state):
        if self.is_leaf:
            return self.mini(state)
        else:
            sub_out = [node(state) for node in self.sub_nodes]
            # Weighted aggregation (simple coherence weighting can be added later)
            return torch.stack(sub_out).mean(dim=0)

class UniversalField(nn.Module):
    def __init__(self, dim=2, damping=0.05):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(dim))
        self.damping = damping

    def update(self, local_ds, coherences, t):
        weighted = torch.stack([c * d for c, d in zip(coherences, local_ds)])
        mean_d = weighted.mean(dim=0)
        phase = torch.sin(torch.tensor(t * 0.05))
        
        self.U.data = (
            (1 - self.damping) * self.U.data +
            self.damping * mean_d +
            0.01 * phase
        )
        return self.U

class Coherence(nn.Module):
    def forward(self, d, U):
        d_n = F.normalize(d, p=2, dim=-1)
        U_n = F.normalize(U, p=2, dim=-1)
        return (d_n * U_n).sum(dim=-1)

class MoravianCore(nn.Module):
    def __init__(self, threshold=0.8, min_depth=3):
        super().__init__()
        self.threshold = threshold
        self.min_depth = min_depth
        self.core_state = None

    def forward(self, local_ds, U, global_coh, current_depth):
        if global_coh > self.threshold and current_depth >= self.min_depth and self.core_state is None:
            self.core_state = U.clone()
            return self.core_state, True
        return self.core_state if self.core_state is not None else U, False

class MTFSC(nn.Module):
    def __init__(self, num_nodes=9, max_depth=4):
        super().__init__()
        self.nodes = nn.ModuleList([FractalNode(max_level=max_depth) for _ in range(num_nodes)])
        self.field = UniversalField()
        self.coherence = Coherence()
        self.core = MoravianCore()
        self.memory = []
        self.max_memory = 100
        self.godot = GodotInterface()  # Placeholder

    def forward(self, states, t, depth=0):
        local_ds = [node(s) for node, s in zip(self.nodes, states)]
        coherences = [self.coherence(d, self.field.U) for d in local_ds]
        global_coh = torch.stack(coherences).mean()

        U = self.field.update(local_ds, coherences, t)

        core_output, crystallized = self.core(local_ds, U, global_coh, depth)

        if crystallized:
            self.memory.append(core_output.clone().detach())
            if len(self.memory) > self.max_memory:
                self.memory.pop(0)

        self.godot.update(U, global_coh, crystallized)

        return {
            'U': U,
            'global_coherence': global_coh.item(),
            'core_output': core_output,
            'crystallized': crystallized,
            'memory_size': len(self.memory)
        }

class GodotInterface:
    def update(self, U, global_coh, crystallized):
        status = "CRYSTALLIZED" if crystallized else "FLOWING"
        print(f"Godot Viz | U-norm: {U.norm():.3f} | Coh: {global_coh:.3f} | Status: {status}")

# Demo
if __name__ == "__main__":
    model = MTFSC(num_nodes=9, max_depth=4)
    states = [torch.randn(8) for _ in range(9)]
    
    for t in range(30):
        out = model(states, t, depth=3)
        print(f"t={t:2d} | Global Coh: {out['global_coherence']:.4f} | Crystallized: {out['crystallized']} | Memory: {out['memory_size']}")
