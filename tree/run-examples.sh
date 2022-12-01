#!/bin/sh

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
  printf "example ${example}"
  output_file="/tmp/${example}_output"
  if (cargo mpirun "$@" --verbose -n ${maxnp} --example "${example}" --features mpi > "${output_file}" 2>&1)
  then
    printf "."
    rm -f "${output_file}"
  else
    printf " failed on %d processes.\noutput:\n" ${num_proc}
    cat "${output_file}"
    rm -f "${output_file}"
    num_failed=$((${num_failed} + 1))
    result="failed"
    continue 2
  fi
  printf " ok.\n"
  num_ok=$((${num_ok} + 1))
done

printf "\nexample result: ${result}. ${num_ok} passed; ${num_failed} failed\n\n"
exit ${num_failed}