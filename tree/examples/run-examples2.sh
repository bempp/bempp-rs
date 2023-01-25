#!/bin/sh

set -e

# enable oversubscribing when using newer Open MPI
export OMPI_MCA_rmaps_base_oversubscribe=1

EXAMPLES_DIR="examples"

examples=$(ls ${EXAMPLES_DIR} | sed "s/\\.rs\$//")
num_examples=$(printf "%d" "$(echo "${examples}" | wc -w)")

maxnp=4
printf "running %d examples\n" ${num_examples}

num_ok=0
num_failed=0
result="ok"

for example in ${examples}
do
  printf "example ${example} on 2...${maxnp} processes"
  output_file="/tmp/${example}_output"
  for num_proc in $(seq 2 ${maxnp})
  do
    echo "cargo mpirun \"$@\" --verbose -n ${num_proc} --example \"${example}\" --features \"mpi\" > \"${output_file}\" 2>&1)"
  done
done
