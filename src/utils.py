import scipy.sparse as sps


# gets coordinates of all genes in GTF file, saving it onto a dictionary with
# the gene_id as the key and coordinates
def get_coords(f, sep_main="\t", sep_scnd=";"):
    CHR_POS = 0
    START_POS = 3
    END_POS = 4
    META_POS = 8
    out_map = {}
    with f:
        for line in f:
            coords = []  # coordinates to be saved in [chr, start, end] format
            # skip if comment
            if (line[0] == "#"):
                continue
            cols = line.split(sep_main)
            # skip if not gene
            if (cols[2] != "gene"):
                continue
            coords.append(cols[CHR_POS])
            coords.append(int(cols[START_POS]))
            coords.append(int(cols[END_POS]))

            gene_id = None
            # get gene_id
            for elem in cols[META_POS].split(sep_scnd):
                kv = elem.split()
                if kv[0] == "gene_id":
                    gene_id = kv[1].strip("\"")  # remove quotation marks
                    break
            out_map[gene_id] = coords
    return out_map


info_elems = ["AF", "AFR_AF", "AMR_AF", "EAS_AF", "EUR_AF", "SAS_AF"]


def vcf_to_maf(f, sep_scnd=";", info_elems=info_elems, chr_pre="chr",
               chr_join=":"):
    CHR_POS = 0
    START_POS = 1
    ID_POS = 2
    INFO_POS = 7
    REF_POS = 3
    ALT_POS = 4

    # takes relevant data from VCF and prints it out in tabular format
    def print_col(chrom, start, id_str, ref_allele, alt_allele, af):
        id_str_0 = id_str
        if id_str == ".":
            l_tmp = [chr_pre + chrom, start, ref_allele, alt_allele]
            id_str_0 = chr_join.join(l_tmp)
        # add addendum to name for allele specificity
        else:
            id_str_0 = id_str + "_" + ref_allele + "/" + alt_allele
        str_tpl = [chrom, start, start, id_str_0] + [af[x] for x in info_elems]
        str_tpl = tuple(str_tpl)
        print(("%s\t%s\t%s\t%s" + ("\t%s" * len(info_elems))) % str_tpl)

    with f:
        for line in f:
            if (line[0] == "#"):
                continue
            cols = line.split("\t")
            chrom = cols[CHR_POS]
            start = cols[START_POS]
            id_str = cols[ID_POS]
            ref_allele = cols[REF_POS]
            alt_alleles = cols[ALT_POS]  # can be multiallelic
            info = cols[INFO_POS]
            # list of different alternative alleles
            alts = alt_alleles.split(",")
            n_alleles = len(alts)
            # list of key value pairs for INFO column for each allele
            afs = [{} for _ in range(n_alleles)]

            for k in info_elems:
                for i in range(n_alleles):
                    afs[i][k] = "NA"  # initialize all values as null

            for elem in info.split(sep_scnd):
                kv = elem.split("=")
                if ((len(kv) != 2) or (kv[0] not in info_elems)):
                    continue
                if ("," in kv[1]):
                    vals = kv[1].split(",")
                    # number of values should be same as number of alleles
                    assert len(vals) == n_alleles
                    for i in range(n_alleles):
                        afs[i][kv[0]] = vals[i]
                else:
                    for i in range(n_alleles):
                        afs[i][kv[0]] = kv[1]

            for i in range(n_alleles):
                print_col(chrom, start, id_str, ref_allele, alts[i], afs[i])


def get_range(pos, base_range=1e6, one_base=True):
    return max(int(one_base), pos - base_range), pos + base_range


# return dictionary matching name to idx
def names_to_idx(names):
    out = {}
    for idx, nme in enumerate(names):
        out[nme] = idx
    return out

# takes a pandas_plink object and converts it to a
def sparse_mat_init(G):
    row_to_idx = names_to_idx(G.sample)
    col_to_idx = names_to_idx(G.snp)
    s_mat = sps.dok_matrix(G.shape, dtype=np.float32)
    return s_mat, row_to_idx, col_to_idx
