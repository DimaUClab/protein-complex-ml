#!/bin/bash

echo -e "index\tparameters"
declare -i ps=0

for model_type in "aminobert" "esm2" "protbert";
do
    for filtering in "interfacearea20" "interfacearea50" "interfacearea80" "rescount20" "rescount50" "rescount80";
    do
	for reduction in "targeted" "uniform";
	do
	    for trial in {0..4};
	    do
		echo -e "$ps\t$model_type $filtering $reduction trial_$trial"
		ps+=1
	    done
	done
    done
done
