#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.samplesheet = null
params.outdir      = 'results'
params.limit       = -1
params.reference   = "${projectDir}/assets/tcga_reference.json"
params.quantiles   = "${projectDir}/assets/tcga_quantiles.json"
params.baseline    = "${projectDir}/assets/tcga_baseline.json"
params.tiles       = 30
params.scaling     = 'p10-p90'
params.show        = 4

process STAIN_NORM {
    tag "$name"
    publishDir "${params.outdir}/fits", mode: 'copy', pattern: '*_fit.json'
    publishDir "${params.outdir}/previews", mode: 'copy', pattern: '*.png'

    input:
    tuple val(name), path(slide)
    path reference
    path quantiles

    output:
    path "${name}.tsv", emit: tsv
    path "${name}_fit.json", optional: true
    path "${name}_*.png", optional: true

    script:
    """
    stain_norm.py --slide ${slide} --name "${name}" --reference ${reference} --quantiles ${quantiles} --scaling ${params.scaling} --tiles ${params.tiles} --show ${params.show}
    """
}

process SUMMARY {
    publishDir params.outdir, mode: 'copy'

    input:
    path tsvs, stageAs: 'tsv/*'
    path baseline

    output:
    path 'stain_norm_results.tsv'
    path 'stain_norm_summary.txt'
    path '*.png', optional: true
    path 'index.html'

    script:
    """
    summarise.py ${baseline}
    """
}

workflow {
    if (!params.samplesheet) {
        error "Please give --samplesheet"
    }
    slides = Channel.fromPath(params.samplesheet, checkIfExists: true)
        .splitCsv(header: true)
        .map { row -> tuple(row.name, file(row.Address)) }
    def limit = params.limit.toString().toInteger()
    if (limit > 0) {
        slides = slides.take(limit)
    }

    STAIN_NORM(slides, file(params.reference, checkIfExists: true), file(params.quantiles, checkIfExists: true))
    SUMMARY(STAIN_NORM.out.tsv.collect(), file(params.baseline, checkIfExists: true))
}
