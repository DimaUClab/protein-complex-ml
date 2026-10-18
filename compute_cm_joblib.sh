#!/usr/bin/env bash


for reduction_type in "interfacearea20" "interfacearea50" "interfacearea80" "rescount20" "rescount50" "rescount80";
do
    for filtering in "targeted" "uniform";
    do
	parallel -j 3 python $PROJ_SCRIPTS/compute_contact_map_joblib.py --generated-proteins "$PROJ_DATASETS/intermediates/generated_proteins/{}/esmc-val-$reduction_type-$filtering-generated_proteins.joblib" --out-file "$PROJ_DATASETS/intermediates/generated_proteins/{}/esmc-val-$reduction_type-$filtering-generated_cm.joblib" --generated-embeddings "$PROJ_DATASETS/intermediates/generated_proteins/{}/esmc-600m-2024-12/esmc-val-$reduction_type-$filtering-generated_embeddings.joblib" --replicate {} ::: 1 2 3
    done
done
