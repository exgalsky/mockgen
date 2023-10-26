import jax
import sys
import os
import gc
import mockgen.defaults as mgd
import xgutil.log_utils as xglogutil
from time import time

import jax.numpy as jnp 
import jax.random as rnd

class Sky:
    '''Sky'''
    def __init__(self, **kwargs):

        self.ID       = kwargs.get(      'ID',mgd.ID)
        self.N        = kwargs.get(       'N',mgd.N)
        self.seed     = kwargs.get(    'seed',mgd.seed)
        self.input    = kwargs.get(   'input',mgd.input)
        self.laststep = kwargs.get('laststep',mgd.laststep)

    def generate(self, **kwargs):

        times={'t0' : time()}

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

        if mpiproc == 0:
            print(f'\nRunning Sky.generate for model "{self.ID}" on {nproc} MPI processes\n')

        if not parallel:
            cube = lpt.Cube(N=self.N,partype=None)
        else:
            jax.distributed.initialize()
            cube = lpt.Cube(N=self.N)
        times = xglogutil.profiletime(None, 'initialization', times, comm, mpiproc)
        if self.laststep == 'init':
            return 0

        #### NOISE GENERATION
        delta = cube.generate_noise(seed=self.seed)
        times = xglogutil.profiletime(None, 'noise generation', times, comm, mpiproc)
        if self.laststep == 'noise':
            return 0

        #### NOISE CONVOLUTION TO OBTAIN DELTA
        delta = cube.noise2delta(delta)
        times = xglogutil.profiletime(None, 'noise convolution', times, comm, mpiproc)
        if self.laststep == 'convolution':
            return 0

        #### LPT DISPLACEMENTS FROM EXTERNAL (WEBSKY AT 768^3) DENSITY CONTRAST
        cube.slpt(infield=self.input,delta=delta)
        times = xglogutil.profiletime(None, '2LPT', times, comm, mpiproc)
        if self.laststep == 'lpt':
            return 0
    
        # # LPT displacements are now in
        # #   cube.s1x
        # #   cube.s1y
        # #   cube.s1z
        # # and
        # #   cube.s2x
        # #   cube.s2y
        # #   cube.s2z

        if mpiproc == 0:
            print(f'Mock sky pipeline after 2LPT is not yet implemented, returning...\n')
        return 0

