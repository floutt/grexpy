import utils
from sklearn.linear_model import ElasticNetCV, LassoCV


def fit_genotypes_enet(geno_mat, pheno_vec, n_jobs=1):
    model = ElasticNetCV(cv=5, random_state=0, n_jobs=n_jobs)
    model.fit(geno_mat, pheno_vec)
    return model


def fit_genotypes_lasso(geno_mat, pheno_vec, n_jobs=1):
    model = LassoCV(cv=5, random_state=0, n_jobs=n_jobs)
    model.fit(geno_mat, pheno_vec)
    return model


def get_gene_cis_region(gene, G, coord_map, base_range, one_base):
    chrm, pos, _ = coord_map[gene]
    beg, end = utils.get_range(pos, base_range, one_base)
    G0 = G.where((G.chrom == chrm) & (G.pos >= beg) & (G.pos < end) &
                 (G.var(axis=0) != 0), drop=True)
    return G0


def fit_all_genes(exp_mat, G, coord_map, fit_method, smat, row_to_idx,
                  col_to_idx, base_range=int(1e6), n_jobs=1, one_base=True):
    fit_fn = None
    if fit_method == "enet":
        fit_fn = fit_genotypes_enet
    elif fit_method == "lasso":
        fit_fn = fit_genotypes_lasso
    else:
        raise ValueError("fit_method must either be enet or lasso")

    for gene in exp_mat.index:
        G0 = get_gene_cis_region(gene, G, coord_map, base_range, one_base)
        model = fit_fn(G0.values, exp_mat.loc[gene])
        var_names = G0.snp.data[model.coef_ != 0]
        var_coef = model.coef_[model.coef_ != 0]
        for name, coef in zip(var_names, var_coef):
            smat[row_to_idx[gene], col_to_idx[name]] = coef
