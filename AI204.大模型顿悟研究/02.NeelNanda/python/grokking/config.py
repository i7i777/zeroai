"""Configuration dataclasses mirroring the experimental setups of
Nanda et al. 2023 (arXiv:2301.05217).

Mainline experiment (paper Section 3):
    P = 113, one-layer ReLU transformer, token embeddings d_model = 128,
    learned positional embeddings, 4 attention heads of dim d_head = 32,
    MLP hidden units n_mlp = 512, no LayerNorm, embed/unembed not tied.
    Train on 30% of the 113*113 pairs, full-batch AdamW(lr=1e-3, wd=1),
    40,000 epochs, loss is cross-entropy read off the final ('=') token.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class GrokConfig:
    # ------- task / data -------
    p: int = 113                     # modulus (prime in the paper)
    frac_train: float = 0.3          # fraction of the P*P pairs used for training

    # ------- architecture -------
    d_model: int = 128
    n_heads: int = 4
    d_head: int = 32
    n_mlp: int = 512
    n_layers: int = 1
    init_std: float = 0.02           # std of the token / positional embeddings

    # ------- optimisation -------
    lr: float = 1e-3
    weight_decay: float = 1.0        # lambda in the paper (AdamW decoupled)
    n_epochs: int = 40_000

    # ------- run / logging -------
    seed: int = 0
    device: str = "cpu"
    log_every: int = 1               # store train/test loss rows every N epochs
    grid_every: int = 200            # store a full P*P*P logit grid every N epochs
    save_grids: bool = True          # keep logit grids on disk (for progress measures)
    out_dir: str = "results"         # directory for run outputs
    tag: str = ""                    # free-form experiment tag (appended to dir name)

    # ---- visualizer snapshots ----
    viz_every: int = 0               # 0 = off; else export a weight snapshot every N epochs
    viz_root: str = ""               # where to write snapshots (default: ../web/public/viz-data)

    # ---- book-keeping ----
    _extra: Dict[str, Any] = field(default_factory=dict, repr=False)

    def run_name(self) -> str:
        """Human readable name for the run folder, unique per experiment."""
        parts = [f"p{self.p}", f"frac{self.frac_train:g}", f"wd{self.weight_decay:g}",
                 f"l{self.n_layers}", f"d{self.d_model}", f"h{self.n_heads}x{self.d_head}",
                 f"m{self.n_mlp}", f"ep{self.n_epochs}"]
        if self.tag:
            parts.insert(0, self.tag)
        return "_".join(parts)

    def out_path(self) -> Path:
        base = Path(self.out_dir) / self.run_name()
        return base / f"seed{self.seed}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("_extra", None)
        d.update(self._extra)
        return d

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GrokConfig":
        known = {f.name for f in fields(cls) if f.name != "_extra"}
        extra = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        cfg = cls(**kwargs)
        cfg._extra = extra
        return cfg


def mainline(p: int = 113, seed: int = 0, n_epochs: Optional[int] = None, **kw) -> GrokConfig:
    """The mainline experiment of the paper."""
    if n_epochs is not None:
        kw["n_epochs"] = n_epochs
    return GrokConfig(p=p, seed=seed, **kw)


def load_config(path: Path) -> GrokConfig:
    return GrokConfig.from_dict(json.loads(Path(path).read_text()))
