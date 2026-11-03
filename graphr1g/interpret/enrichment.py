"""Pre-ranked enrichment over explicitly restricted node universes."""
def prerank(scores, hallmark, genes, seed, permutations=1000):
    import gseapy
    universe = set(genes)
    sets = {name: sorted(set(members) & universe) for name, members in hallmark.items()}
    sets = {name: members for name, members in sets.items() if 10 <= len(members) <= 500}
    ranking = scores[scores["gene"].isin(universe)][["gene", "score"]].drop_duplicates("gene").sort_values("score", ascending=False)
    if not sets or ranking.empty:
        raise ValueError("No eligible pathways or ranked genes")
    return gseapy.prerank(rnk=ranking, gene_sets=sets, permutation_num=permutations, min_size=10, max_size=500, seed=seed, outdir=None, threads=1).res2d
