# mockgen
Interface to GPU-enabled python library for generating extragalactic mock skies.

## Installation
1. git clone https://github.com/exgalsky/mockgen.git
2. cd mockgen
3. pip install .

## Running
Currently runs on perlmutter in the [xgsmenv](https://github.com/exgalsky/xgsmenv) enviroment.

Mocks can be generated through the command line interface:
```
# on Perlmutter at NERSC with one GPU node:
% module use /global/cfs/cdirs/mp107/exgal/env/xgsmenv/20231013-0.0.0/modulefiles/
% module load xgsmenv
% salloc -N 1 -C gpu
% export XGSMENV_NGPUS=4
% srun -n 4 xgmockgen "mockgen test-512" --N 512 --seed 13579 --ityp delta

Running Sky.generate for model "mockgen test-512" on 4 MPI processes

1.998147 sec for initialization

2.314237 sec for noise generation

2.649199 sec for noise convolution

6.202100 sec for 2LPT

Mock sky pipeline after 2LPT is not yet implemented, returning...

```
