#!/bin/bash
contact_map_dir="$PROJ_ROOT/experiments/protein-adaptive_sampling/contact_map_predictions"
out_dir="$PROJ_DATASETS/intermediates/similarity_data/similarities"
while read -r psA mA reductionCriterionA reductionTypeA trialA psB mB reductionCriterionB reductionTypeB trialB;
do
    #echo "$psA $mA $reductionCriterionA $reductionTypeA $trialA $psB $mB $reductionCriterionB $reductionTypeB $trialB"
    #mA_holdout_file="$contact_map_dir/$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz"
    #mB_holdout_file="$contact_map_dir/$mB-$reductionCriterionB-$reductionTypeB-$trialB-predictions-ps$psB.npz"
    
    #python ${PROJ_SCRIPTS}/compute_cmp_similarities.py -A "$mA_holdout_file" -B "$mB_holdout_file" --output "$out_dir/$mA-$mB-$reductionCriterionA-$reductionTypeA-$trialA-holdout-similarities.dat" --set holdout --parameters ${PROJ_SCRIPTS}/parameter_sets.txt --index "$psA"

    mA_val_file="$contact_map_dir/$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz"
    mB_val_file="$contact_map_dir/$mB-$reductionCriterionB-$reductionTypeB-$trialB-predictions-ps$psB.npz"
    
    python ${PROJ_SCRIPTS}/compute_cmp_similarities.py -A "$mA_val_file" -B "$mB_val_file" --output "$out_dir/$mA-$mB-$reductionCriterionA-$reductionTypeA-$trialA-val-similarities.dat" --set val --parameters ${PROJ_SCRIPTS}/parameter_sets.txt --index "$psA"

    
    mA_train_file="$contact_map_dir/$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz"
    mB_train_file="$contact_map_dir/$mB-$reductionCriterionB-$reductionTypeB-$trialB-predictions-ps$psB.npz"
    
    python ${PROJ_SCRIPTS}/compute_cmp_similarities.py -A "$mA_train_file" -B "$mB_train_file" --output "$out_dir/$mA-$mB-$reductionCriterionA-$reductionTypeA-$trialA-train-similarities.dat" --set train --parameters ${PROJ_SCRIPTS}/parameter_sets.txt --index "$psA"
done < ${PROJ_SCRIPTS}/cmp_aminobert_esm2.txt
