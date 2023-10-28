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
% srun -n 4 mockgen mockgen-test

Running Sky.generate for model "mockgen-test" on 4 MPI processes

13.234693 sec for initialization

3.070061 sec for noise generation

5.516870 sec for noise convolution

5.704113 sec for 2LPT

28.663420 sec for field mapping

```
