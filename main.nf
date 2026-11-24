#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

params.samplesheet = null
params.outdir      = 'results'
params.limit       = -1
params.reference   = "${projectDir}/assets/tcga_reference.json"
params.tiles       = 30

process STAIN_NORM {
    tag "$name"
    publishDir "${params.outdir}/fits", mode: 'copy', pattern: '*_fit.json'
    publishDir "${params.outdir}/previews", mode: 'copy', pattern: '*.png'

    input:
    tuple val(name), path(slide)
    path reference

    output:
    path "${name}.tsv", emit: tsv
    path "${name}_fit.json", optional: true
    path "${name}.png", optional: true

    script:
    """
    stain_norm.py --slide "${slide}" --name "${name}" --reference ${reference} --tiles ${params.tiles}
    """
}

process SUMMARY {
    publishDir params.outdir, mode: 'copy'

    input:
    path tsvs, stageAs: 'tsv/*'

    output:
    path 'stain_norm_results.tsv'
    path 'stain_norm_summary.txt'
    path 'stain_norm_summary.png', optional: true
    path 'index.html'

    script:
    """
    summarise.py
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

    STAIN_NORM(slides, file(params.reference, checkIfExists: true))
    SUMMARY(STAIN_NORM.out.tsv.collect())
}
