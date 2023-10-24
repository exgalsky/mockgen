import jax
import sys
import os
import gc
import mockgen.util as mockutil
from time import time

import jax.numpy as jnp 
import jax.random as rnd

class Sky:
    '''Sky'''
    def __init__(self, **kwargs):

        self.ID   = kwargs.get(  'ID',mockutil.MockgenDefaults['ID'])
        self.N    = kwargs.get(   'N',mockutil.MockgenDefaults['N'])
        self.seed = kwargs.get('seed',mockutil.MockgenDefaults['seed'])
        self.ityp = kwargs.get('ityp',mockutil.MockgenDefaults['ityp'])

    def generate(self, **kwargs):

        times={'t0' : time()}

        mockutil.parprint(f'Running Sky.generate for model {self.ID}')

        import mockgen
        import jax
        import lpt
        from mpi4py import MPI
        jax.config.update("jax_enable_x64", True)

        parallel = False
        nproc    = MPI.COMM_WORLD.Get_size()
        mpiproc  = MPI.COMM_WORLD.Get_rank()
        comm     = MPI.COMM_WORLD
        task_tag = "MPI process "+str(mpiproc)

        if MPI.COMM_WORLD.Get_size() > 1: parallel = True

        return 0

        # if not parallel:
        #     cube = lpt.Cube(N=N,partype=None)
        # else:
        #     jax.distributed.initialize()
        #     cube = lpt.Cube(N=N)
        # times = mockutil.profiletime(None, 'initialization', times, comm, mpiproc)

        # #### NOISE GENERATION
        # delta = cube.generate_noise(seed=seed)
        # times = mockutil.profiletime(None, 'noise generation', times, comm, mpiproc)

        # #### NOISE CONVOLUTION TO OBTAIN DELTA
        # delta = cube.noise2delta(delta)
        # times = mockutil.profiletime(None, 'noise convolution', times, comm, mpiproc)

        # #### 2LPT DISPLACEMENTS FROM EXTERNAL (WEBSKY AT 768^3) DENSITY CONTRAST
        # cube.slpt(infield=ityp,delta=delta)
        # times = mockutil.profiletime(None, '2LPT', times, comm, mpiproc)

        # # LPT displacements are now in
        # #   cube.s1x
        # #   cube.s1y
        # #   cube.s1z
        # # and
        # #   cube.s2x
        # #   cube.s2y
        # #   cube.s2z

        # return 0

