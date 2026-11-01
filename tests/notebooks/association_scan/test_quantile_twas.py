"""Notebook tier: quantile TWAS end to end on the qtl_mini / twas / ld_reference fixtures.

Chain (the production entrypoints, one ``sos run`` per gene as on the cluster):
  qr_and_twas.ipynb quantile_qtl_twas_weight  (per gene, per-gene cwd; 2 contexts)
    -> xqtl meta table (#chr region_id TSS start end contexts original_data)
    -> twas_ctwas.ipynb quantile_twas  (one LD block, chr22_10000000_19000000)

Exercises: the [get_analysis_regions] -> [qtl_dataset_construct] -> weight chain with two
phenotype/covariate pairs (multi-context aggregation), qQTLR::extract_qtl_genotypes /
get_context_phenotypes / get_context_covariates / parse_region, the legacy LD-reference
variant filter, and the restored [get_quantile_twas_regions] provider feeding
[quantile_twas] (qQTLR::load_quantile_twas_weights + quantile_twas_pipeline).

Fixtures: qtl_mini phenotypes (example_geneexpr{,_ctx2}.bed.gz, QTLtools-format covariates;
49 samples), the 60-sample ld_prune_reference chr22 PLINK1 genotypes (same SAMPLE_* IDs; the
qtl_mini example.chr22 genotypes share no variants with the LD panel / GWAS), the chr22
synthetic GWAS (twas/), and the chr22 pgen LD panel (ld_reference/). The pheno and GWAS metas
carry repo-relative paths, so absolute-path copies are written into tmp; the LD meta is used in
place, because genotype-format LD panels are resolved relative to the meta file.

Also covers the "*" (spanning deletion) alleles in the genotypes, which used to crash
allele_qc() in [quantile_twas].
"""
from __future__ import annotations

import csv
import gzip
import json
import subprocess

import pytest

from helpers.r_runner import rscript_bin

QR_NB = "pipeline/qr_and_twas.ipynb"
TWAS_NB = "pipeline/twas_ctwas.ipynb"
FIX = "tests/fixtures"
GENES = ["ENSG00000283047", "ENSG00000130538"]
CONTEXTS = ["fixture_ctx1", "fixture_ctx2"]
LD_BLOCK = "chr22_10000000_19000000"

# QR screen settings. Production uses fdr / 0.05, but the 49-sample fixture has no SNP below
# BH 0.05, so every SNP is passed through the screen (the test is <, so 1.01 keeps all) to
# exercise the LD / MAF / correlation filters, weight fitting and TWAS downstream.
SCREEN_METHOD = "fdr"
SCREEN_THRESHOLD = 1.01

TWAS_COLUMNS = ["chr", "molecular_id", "TSS", "start", "end", "context", "gwas_study",
                "method", "quantile_start", "quantile_end", "pseudo_R2_avg",
                "twas_z", "twas_pval", "type", "block"]


def _abs_paths(src, dst, col, base):
    """Copy a TSV meta, rewriting column ``col`` (0-based, relative to ``base``) to an absolute path."""
    with open(src) as fi, open(dst, "w", newline="") as fo:
        rows = list(csv.reader(fi, delimiter="\t"))
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(rows[0])
        for r in rows[1:]:
            r[col] = str(base / r[col])
            w.writerow(r)
    return dst


def _weight_rds(cwd, gene):
    hits = sorted((cwd / "quantile_qtl_twas_weight").glob(f"fixture.chr22_{gene}.univariate_qr_twas_weights.rds"))
    return hits[0] if hits else None


def _weight_contexts(path):
    """{gene: [context entry names]} of a univariate_qr_twas_weights RDS, plus which entries
    carry TWAS weights (``twas_variant_names``) — via a small Rscript (rds_probe.R only
    reports top-level names)."""
    expr = (
        "suppressMessages(library(jsonlite)); x <- readRDS(commandArgs(TRUE)[1]); "
        "cat(toJSON(lapply(x, function(g) lapply(g, function(e) "
        "list(has_weights = is.list(e) && 'twas_variant_names' %in% names(e)))), auto_unbox = TRUE))"
    )
    p = subprocess.run([rscript_bin(), "-e", expr, str(path)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(p.stdout)


def _twas_rows(path):
    with gzip.open(path, "rt") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


@pytest.fixture(scope="module")
def quantile_chain(run_sos, repo_root, tmp_path_factory):
    """Run the weight step for both genes, write the xqtl meta, run quantile_twas once.
    Returns the paths the tests assert on."""
    fx = repo_root / FIX
    qm = fx / "qtl_mini"
    d = tmp_path_factory.mktemp("quantile_twas")
    inp = d / "inputs"
    inp.mkdir()

    pheno = [_abs_paths(qm / "pheno_manifest.tsv", inp / "pheno_ctx1.region_list.txt", 4, qm),
             _abs_paths(qm / "pheno_manifest_ctx2.tsv", inp / "pheno_ctx2.region_list.txt", 4, qm)]
    ld_meta = fx / "ld_reference/ld_meta_file.tsv"
    gwas_meta = inp / "gwas_meta.tsv"
    with open(fx / "twas/protocol_example.twas.gwas_meta.tsv") as fi, open(gwas_meta, "w", newline="") as fo:
        rows = list(csv.reader(fi, delimiter="\t"))
        w = csv.writer(fo, delimiter="\t", lineterminator="\n")
        w.writerow(rows[0])
        for r in rows[1:]:
            r[2] = str(repo_root / r[2])                 # repo-relative -> absolute
            w.writerow(r)
    type_table = inp / "data_type_table.txt"
    type_table.write_text("type\tcontext\n" + "".join(f"eQTL\t{c}\n" for c in CONTEXTS))

    # (1) TWAS weights, one sos call per gene with its own cwd (as production does).
    weights = {}
    for gene in GENES:
        cwd = d / "weights" / gene
        p = run_sos(
            repo_root / QR_NB, "quantile_qtl_twas_weight",
            {
                "name": "fixture",
                "genoFile": fx / "ld_prune_reference/genotype/protocol_example.ld_genotype.chr22.bed",
                "phenoFile": pheno,
                "covFile": [qm / "example_covariates.tsv"] * 2,
                "customized-association-windows": qm / "association_windows.bed",
                "phenotype-names": CONTEXTS,
                "screen_method": SCREEN_METHOD,
                "screen_threshold": SCREEN_THRESHOLD,
                "no-marginal-beta-calculate": True,
                "no-vqtl-calculate": True,
                "ld_reference_meta_file": ld_meta,
                "region-name": gene,
                "cwd": cwd,
            },
            cwd=repo_root, timeout=1800)
        assert p.returncode == 0, p.stdout + p.stderr
        weights[gene] = _weight_rds(cwd, gene)

    # (2) xqtl meta: TSS = phenotype start, start/end = association window.
    tss = {r[3]: r[1] for r in csv.reader(open(qm / "pheno_manifest.tsv"), delimiter="\t") if not r[0].startswith("#")}
    xqtl_meta = inp / "xqtl_meta.tsv"
    with open(xqtl_meta, "w") as fo:
        fo.write("#chr\tregion_id\tTSS\tstart\tend\tcontexts\toriginal_data\n")
        for r in csv.reader(open(qm / "association_windows.bed"), delimiter="\t"):
            if r[0].startswith("#") or weights.get(r[3]) is None:
                continue
            fo.write(f"{r[0]}\t{r[3]}\t{tss[r[3]]}\t{r[1]}\t{r[2]}\tNA\t{weights[r[3]]}\n")

    # (3) quantile TWAS over the one fixture LD block.
    twas_cwd = d / "twas"
    p = run_sos(
        repo_root / TWAS_NB, "quantile_twas",
        {
            "gwas_meta_data": gwas_meta,
            "ld_meta_data": ld_meta,
            "region_name": LD_BLOCK,
            "xqtl_meta_data": xqtl_meta,
            "xqtl_type_table": type_table,
            "name": "fixture",
            "cwd": twas_cwd,
        },
        cwd=repo_root, timeout=1800)
    assert p.returncode == 0, p.stdout + p.stderr
    out = twas_cwd / "quantile_twas"
    return {
        "weights": weights,
        "twas_tsv": out / f"fixture.{LD_BLOCK}.quantile_twas.tsv.gz",
        "twas_data": out / f"fixture.{LD_BLOCK}.quantile_twas_data.rds",
        "grouped": out / f"fixture.{LD_BLOCK}.grouped_quantile_twas_weight.rds",
    }


@pytest.mark.parametrize("gene", GENES)
def test_quantile_weights(quantile_chain, gene):
    rds = quantile_chain["weights"][gene]
    assert rds is not None and rds.exists(), f"no weight RDS for {gene}"
    summ = _weight_contexts(rds)
    assert list(summ) == [gene]
    # Entry names are "<context>_<gene>" (context != trait ID), one per context.
    entries = list(summ[gene])
    assert len(entries) == len(CONTEXTS), entries
    for ctx in CONTEXTS:
        assert any(e.startswith(ctx) for e in entries), (ctx, entries)
    assert all(v["has_weights"] for v in summ[gene].values()), summ


def test_quantile_twas(quantile_chain, read_rds):
    tsv = quantile_chain["twas_tsv"]
    assert tsv.exists(), tsv
    rows = _twas_rows(tsv)
    assert rows, "empty quantile TWAS table"
    assert list(rows[0]) == TWAS_COLUMNS
    assert {r["block"] for r in rows} == {LD_BLOCK}
    assert {r["molecular_id"] for r in rows} <= set(GENES)
    assert {r["type"] for r in rows} == {"eQTL"}
    assert quantile_chain["twas_data"].exists()
    assert read_rds(quantile_chain["grouped"])["class"] == "list"
    # Pinned to the run that is bit-identical between pecotmr 0.6.11 and 0.9.4 + qQTLR.
    assert len(rows) == 33
    assert {(g, c) for g in GENES for c in CONTEXTS} == {(r["molecular_id"], r["context"]) for r in rows}
    top = min(rows, key=lambda r: float(r["twas_pval"]))
    assert (top["molecular_id"], top["context"], top["method"]) == ("ENSG00000283047", "fixture_ctx2", "A3")
    assert float(top["twas_z"]) == pytest.approx(3.91282400197658, abs=1e-6)
