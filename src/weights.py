import utils
import pickle
from sklearn.linear_model import ElasticNetCV, LassoCV
import numpy as np
import scipy.sparse as sps
import pandas as pd


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
        model = fit_fn(G0.values, exp_mat.loc[gene], n_jobs=n_jobs)
        var_names = G0.snp.data[model.coef_ != 0]
        var_coef = model.coef_[model.coef_ != 0]
        for name, coef in zip(var_names, var_coef):
            smat[row_to_idx[gene], col_to_idx[name]] = coef


def load_weight_matrix(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class WeightMatrix:
    def __init__(self, G, pheno, coord_map, fit_method, base_range=int(1e6),
                 one_base=True, n_jobs=1):
        self._row_to_idx = self._names_to_idx(pheno.index)
        self._col_to_idx = self._names_to_idx(G.snp.data)
        self._rownames = pheno.index
        self._colnames = G.snp.data
        self._smat = sps.dok_matrix((len(pheno.index), len(G.snp)),
                                    dtype=np.float32)
        self._r2 = []
        self._base_range = base_range
        self._one_base = one_base
        self._fit_genes(G, pheno, coord_map, fit_method, n_jobs=n_jobs)

    def _names_to_idx(self, names):
        out = {}
        for idx, nme in enumerate(names):
            out[nme] = idx
        return out

    def _fit_genes(self, G, pheno, coord_map, fit_method, n_jobs=1):
        fit_fn = None
        if fit_method == "enet":
            fit_fn = fit_genotypes_enet
        elif fit_method == "lasso":
            fit_fn = fit_genotypes_lasso
        else:
            raise ValueError("fit_method must either be enet or lasso")

        for gene in pheno.index:
            G0 = get_gene_cis_region(gene, G, coord_map, self._base_range,
                                     self._one_base)
            if G0.shape[1] == 0:
                print("WARNING: no variants found for gene %s" % gene)
                self._r2.append(0)
                continue
            model = fit_fn(G0.values, pheno.loc[gene], n_jobs=n_jobs)
            self._r2.append(model.score(G0.values, pheno.loc[gene]))
            var_names = G0.snp.data[model.coef_ != 0]
            var_coef = model.coef_[model.coef_ != 0]
            for name, coef in zip(var_names, var_coef):
                self._smat[self._row_to_idx[gene],
                           self._col_to_idx[name]] = coef

    def predict(self, G):
        grex_mat = self._smat @ G.values.T
        return(pd.DataFrame(grex_mat, index=self._rownames,
                            columns=G.sample.data))

    def save(self, outfile):
        self._smat = self._smat.tocsr()
        with open(outfile, "wb") as f:
            pickle.dump(self, f)
