#!/bin/bash -l

# Request time (hours:minutes:seconds)
#$ -l h_rt=0:30:0

# Request memory per core
#$ -l mem=2G

# Set name
#$ -N MPI-Test-2

# Request cores
#$ -pe mpi 80

# Set working directory
#$ -wd /home/ucapska/Scratch

# Run the job
gerun $HOME/bempp-rs/target/release/examples/uniform_scaling