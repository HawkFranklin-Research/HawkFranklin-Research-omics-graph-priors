"""Graph artifacts shared by the v2 models."""
from dataclasses import dataclass, field


@dataclass
class GraphArtifact:
    edge_index: object
    edge_weight: object
    name: str
    fold: int | None = None
    seed: int | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def edge_attr(self):
        return self.edge_weight.reshape(-1, 1)
