#!/bin/bash
while read -r psA mA reductionCriterionA reductionTypeA trialA psB mB reductionCriterionB reductionTypeB trialB;
do
    echo "$psA $mA $reductionCriterionA $reductionTypeA $trialA $psB $mB $reductionCriterionB $reductionTypeB $trialB"
    mA_train_file="$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz.sampled-train_data_sampled_0.25.npz"
    mB_train_file="$mB-$reductionCriterionB-$reductionTypeB-$trialB-predictions-ps$psB.npz.sampled-train_data_sampled_0.25.npz"
    
    train_keys="$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz.sampled-train_data_sampled_keys_0.25.dat"
    
    python compute_cmp_similarities_keys.py -A "$mA_train_file" -B "$mB_train_file" -k "$train_keys" --meta dips_plus_all.csv --output "$mA-$mB-$reductionCriterionA-$reductionTypeA-$trialA-train-similarities_0.25.dat" --set train

    #

    # mA_val_file="$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz.sampled-val_data_sampled.npz"
    # mB_val_file="$mB-$reductionCriterionB-$reductionTypeB-$trialB-predictions-ps$psB.npz.sampled-val_data_sampled.npz"
    
    
    # val_keys="$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz.sampled-val_data_sampled_keys.dat"
    
    # python compute_cmp_similarities_keys.py -A "$mA_val_file" -B "$mB_val_file" -k "$val_keys" --meta dips_plus_all.csv --output "$mA-$mB-$reductionCriterionA-$reductionTypeA-$trialA-val-similarities.dat" --set val

    # mA_holdout_file="$mA-$reductionCriterionA-$reductionTypeA-$trialA-predictions-ps$psA.npz"
    # mB_holdout_file="$mB-$reductionCriterionB-$reductionTypeB-$trialB-predictions-ps$psB.npz"
    
    # python compute_cmp_similarities_keys.py -A "$mA_holdout_file" -B "$mB_holdout_file" --meta dips_plus_all.csv --output "$mA-$mB-$reductionCriterionA-$reductionTypeA-$trialA-holdout-similarities.dat" -k holdout_val_keys.dat --set holdout
done < cmp_aminobert_esm2.txt
