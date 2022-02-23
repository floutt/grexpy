import utils
from sklearn.linear_model import ElasticNet, Lasso
from pandas_plink import read_plink1_bin


def fit_genotypes_enet(geno_mat, pheno_vec):
    model = ElasticNet()
    model.fit(geno_mat, pheno_vec)
    return model


def fit_genotypes_lasso(geno_mat, pheno_vec):
    model = Lasso(alpha=0.5)
    model.fit(geno_mat, pheno_vec)
    return model


def get_gene_cis_region(gene, G, coord_map, base_range, one_base):
    _, pos, _ = coord_map[gene]
    beg, end = utils.get_range(pos, base_range, one_base)
    G0 = G.where((G.pos >= beg) & (G.pos < end), drop=True)
    return G0
