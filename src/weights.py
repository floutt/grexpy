import utils
import pickle
from sklearn.linear_model import ElasticNetCV, LassoCV
from math import nan
import numpy as np
import scipy.sparse as sps
import pandas as pd
import copy


def fit_genotypes_enet(geno_mat, pheno_vec, n_jobs=1):
    model = ElasticNetCV(cv=5, random_state=0, n_jobs=n_jobs, max_iter=10000)
    model.fit(geno_mat, pheno_vec)
    return model


def fit_genotypes_lasso(geno_mat, pheno_vec, n_jobs=1):
    model = LassoCV(cv=5, random_state=0, n_jobs=n_jobs, max_iter=10000)
    model.fit(geno_mat, pheno_vec)
    return model


def get_gene_cis_region(gene, G, coord_map, base_range, one_base):
    chrm, pos, _ = coord_map[gene]
    beg, end = utils.get_range(pos, base_range, one_base)
    G0 = G.where((G.chrom == chrm) & (G.pos >= beg) & (G.pos < end),
                 drop=True)
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
        G = G.assign_coords(var_geno=("variant", G.values.var(axis=0)))
        G = G.where(np.logical_and(np.invert(G.var_geno.isnull()),
                                   G.var_geno != 0), drop=True)
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
                self._r2.append(nan)
                continue
            model = fit_fn(G0.values, pheno.loc[gene], n_jobs=n_jobs)
            self._r2.append(model.score(G0.values, pheno.loc[gene]))
            var_names = G0.snp.data[model.coef_ != 0]
            var_coef = model.coef_[model.coef_ != 0]
            for name, coef in zip(var_names, var_coef):
                self._smat[self._row_to_idx[gene],
                           self._col_to_idx[name]] = coef

    def __getitem__(self, key):
        out_obj = copy.copy(self)
        x, y = key
        row_idx = []
        col_idx = []
        if isinstance(x, str):
            x = [x]
        if isinstance(y, str):
            y = [y]

        idx_error_msg = "Indices should be of type str or a list of strings"
        for x0 in x:
            try:
                assert isinstance(x0, str)
                row_idx.append(self._row_to_idx[x0])
            except KeyError:
                raise KeyError("Value %s not a row in WeightMatrix" % x0)
            except AssertionError:
                raise ValueError(idx_error_msg)

        for y0 in y:
            try:
                assert isinstance(y0, str)
                col_idx.append(self._col_to_idx[y0])
            except KeyError:
                raise KeyError("Value %s not a column in WeightMatrix" % y0)
            except AssertionError:
                raise ValueError(idx_error_msg)

        out_obj._rownames = [out_obj._rownames[idx] for idx in row_idx]
        out_obj._colnames = [out_obj._colnames[idx] for idx in col_idx]
        out_obj._r2 = [out_obj._r2[idx] for idx in row_idx]
        out_obj._row_to_idx = self._names_to_idx(out_obj._rownames)
        out_obj._col_to_idx = self._names_to_idx(out_obj._colnames)
        out_obj._smat = self._smat[row_idx, col_idx]
        return out_obj

    def predict(self, G):
        grex_mat = self._smat @ G.values.T
        return(pd.DataFrame(grex_mat, index=self._rownames,
                            columns=G.sample.data))

    def save(self, outfile):
        self._smat = self._smat.tocsr()
        with open(outfile, "wb") as f:
            pickle.dump(self, f)
