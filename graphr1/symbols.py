"""Resolve historical gene symbols against an HGNC reference table."""
from collections import defaultdict

import pandas as pd


def source_symbols(columns):
    """Expand multi-gene feature columns while retaining their source column."""
    return [(column, part.strip().split("|")[0]) for column in columns
            for part in column.removeprefix("GENE:").split(";") if part.strip()]


class HGNCHarmonizer:
    def __init__(self, hgnc_df, locus_groups=None):
        df = hgnc_df.loc[hgnc_df["status"].eq("Approved")].copy()
        if locus_groups is not None:
            df = df.loc[df["locus_group"].isin(locus_groups)]
        self.index = {key: defaultdict(set) for key in ("approved", "previous", "alias")}
        for row in df.itertuples(index=False):
            target = (str(row.symbol), str(row.hgnc_id))
            self.index["approved"][target[0].upper()].add(target)
            for field, kind in (("prev_symbol", "previous"), ("alias_symbol", "alias")):
                value = getattr(row, field, None)
                if pd.notna(value):
                    for alias in str(value).split("|"):
                        if alias.strip():
                            self.index[kind][alias.strip().upper()].add(target)

    def map(self, symbols):
        records = []
        for source in symbols:
            source = str(source).strip()
            matches, method = set(), "unmapped"
            for kind in ("approved", "previous", "alias"):
                matches = self.index[kind].get(source.upper(), set())
                if matches:
                    method = kind if len(matches) == 1 else "ambiguous"
                    break
            approved, identifier = next(iter(matches)) if len(matches) == 1 else (None, None)
            records.append(dict(source=source, approved=approved, hgnc_id=identifier, method=method))
        return pd.DataFrame(records, columns=["source", "approved", "hgnc_id", "method"])
