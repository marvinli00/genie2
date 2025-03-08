#!/bin/bash

# Create the main output directory if it doesn't exist
mkdir -p ./protein_generated

# Outer loop from 1 to 20
for outer in {0..9}; do
    # Create directory for this outer index
    mkdir -p ./protein_generated/$outer
    
    # Inner loop from 1 to 10
    for inner in {1..10}; do
        # Run the Python command with inner loop index
        python genie/sample_unconditional_motif\ copy.py --name test002 --epoch 30 --scale 1 --outdir results/test002 --num_samples 10 --motif_index $inner
        
        # Copy the generated file to the destination
        cp /hai/scratch/mli89/protein_design/genie2/results/test002/pdbs/75_0.pdb ./protein_generated/$outer/$inner.pdb
    done
done