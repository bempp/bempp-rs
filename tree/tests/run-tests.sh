#!/bin/sh

set -e

# enable oversubscribing when using newer Open MPI
export OMPI_MCA_rmaps_base_oversubscribe=1

TEST_DIR="tests"

tests=$(ls ${TEST_DIR} | sed "s/\\.rs\$//")
num_tests=$(printf "%d" "$(echo "${tests}" | wc -w)")

maxnp=4
printf "running %d tests\n" ${num_tests}

num_ok=0
num_failed=0
result="ok"

for test in ${tests}
do
  printf "test ${test} on 2...${maxnp} processes"
  output_file="/tmp/${test}_output"
  for num_proc in $(seq 2 ${maxnp})
  do
    if (cargo mpirun "$@" --verbose -n ${num_proc} --test "${test}" > "${output_file}" 2>&1)
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
  done
  printf " ok.\n"
  num_ok=$((${num_ok} + 1))
done

printf "\ntest result: ${result}. ${num_ok} passed; ${num_failed} failed\n\n"
exit ${num_failed}