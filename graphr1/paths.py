from pathlib import Path


class ExperimentPaths:
    def __init__(self, root, name, cancer=None):
        self.root = Path(root) / name
        if cancer is not None:
            self.root = self.root / cancer

    def stage(self, name):
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def run(self, condition, repeat, fold):
        return self.stage("05_runs") / condition / f"r{repeat}_f{fold}"

    def baseline(self, name, repeat, fold):
        return self.stage("06_baselines") / name / f"r{repeat}_f{fold}"

    def graph(self, name, repeat=None, fold=None):
        name = "string" if name == "none" else name
        root = self.stage("04_graphs") / name
        return root if repeat is None else root / f"r{repeat}_f{fold}"

    def normalization(self, repeat, fold):
        return self.stage("04_graphs") / "normalization" / f"r{repeat}_f{fold}"

    def eval_mask(self, repeat, fold):
        return self.stage("05_runs") / "_eval_masks" / f"r{repeat}_f{fold}.npz"
