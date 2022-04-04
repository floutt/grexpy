import utils
from sklearn.linear_model import ElasticNetCV, LassoCV
from pandas_plink import read_plink1_bin


def fit_genotypes_enet(geno_mat, pheno_vec, n_jobs):
    model = ElasticNetCV(cv=5, random_state=0, n_jobs=n_jobs)
    model.fit(geno_mat, pheno_vec)
    return model


def fit_genotypes_lasso(geno_mat, pheno_vec, n_jobs):
    model = LassoCV(cv=5, random_state=0, n_jobs=n_jobs)
    model.fit(geno_mat, pheno_vec)
    return model


def get_gene_cis_region(gene, G, coord_map, base_range, one_base):
    chrm, pos, _ = coord_map[gene]
    beg, end = utils.get_range(pos, base_range, one_base)
    G0 = G.where((G.chrom == chrm) & (G.pos >= beg) & (G.pos < end) &
                 (G.var(axis=0) != 0), drop=True)
    return G0
