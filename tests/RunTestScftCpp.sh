export KMP_STACKSIZE=1G
export MKL_NUM_THREADS=1  # always 1
export OMP_MAX_ACTIVE_LEVELS=2  # 0, 1 or 2
./Test.out TestInputParams
