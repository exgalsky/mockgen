import sys
import os
import mockgen.defaults as mgd
import xgutil.log_util as xglogutil

def _test_transfer():
    import jax.numpy as jnp
    k  = jnp.logspace(-3,2,1000)
    pk = jnp.sqrt(1e5 * (k/1e-2) * ((1+(k/1e-2)**2)/2)**-4) # something reasonable for testing purposes
    return jnp.asarray([k,pk]).T

class Sky:
    '''Sky'''
    def __init__(self, **kwargs):

        self.ID       = kwargs.get(      'ID',mgd.ID)
        self.seed     = kwargs.get(    'seed',mgd.ID)
        self.N        = kwargs.get(       'N',mgd.N)
        self.Niter    = kwargs.get(   'Niter',mgd.Niter)
        self.input    = kwargs.get(   'input',mgd.input)
        self.Lbox     = kwargs.get(    'Lbox',mgd.Lbox)
        self.laststep = kwargs.get('laststep',mgd.laststep)
        self.Nside    = kwargs.get(   'Nside',mgd.Nside)
        self.gpu      = kwargs.get(     'gpu',mgd.gpu)
        self.mpi      = kwargs.get(     'mpi',mgd.mpi)

        from mpi4py import MPI

        self.parallel = False
        self.nproc    = MPI.COMM_WORLD.Get_size()
        self.mpiproc  = MPI.COMM_WORLD.Get_rank()
        self.comm     = MPI.COMM_WORLD
        self.task_tag = "MPI process "+str(self.mpiproc)

        if MPI.COMM_WORLD.Get_size() > 1: self.parallel = True

    def run(self, **kwargs):
        import jax
        import lpt
        from time import time
        times={'t0' : time()}

        if not self.parallel:
            cube = lpt.Cube(N=self.N,partype=None)
        else:
            jax.distributed.initialize()
            cube = lpt.Cube(N=self.N)
        if self.laststep == 'init':
            return 0
        
        err = 0
        seeds = range(self.seed,self.seed+self.Niter)
        i = 0
        for seed in seeds:
            if i==1:
                times={'t0' : time()}
            err += self.generatesky(seed,cube,times)
            i += 1
        xglogutil.summarizetime(None,times,self.comm, self.mpiproc)
        
        return err

    def generatesky(self, seed, cube, times, **kwargs):
        from time import time
        import datetime

        import jax
        import lpt
        from xgfield import fieldsky
        jax.config.update("jax_enable_x64", True)

        import logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        if self.mpiproc == 0:
            xglogutil.parprint(f'\nGenerating sky for model "{self.ID}" with seed={seed}')

        #### NOISE GENERATION
        delta = cube.generate_noise(seed=seed)
        times = xglogutil.profiletime(None, 'noise generation', times, self.comm, self.mpiproc)
        if self.laststep == 'noise':
            return 0

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        #### NOISE CONVOLUTION TO OBTAIN DELTA
        delta = cube.noise2delta(delta,_test_transfer())
        times = xglogutil.profiletime(None, 'noise convolution', times, self.comm, self.mpiproc)
        if self.laststep == 'convolution':
            return 0

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
                
        #### LPT DISPLACEMENTS FROM DENSITY CONTRAST
        cube.slpt(infield=self.input,delta=delta)
        times = xglogutil.profiletime(None, '2LPT', times, self.comm, self.mpiproc)
        if self.laststep == 'lpt':
            return 0

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    
        # # LPT displacements are now in
        # #   cube.s1x
        # #   cube.s1y
        # #   cube.s1z
        # # and
        # #   cube.s2x
        # #   cube.s2y
        # #   cube.s2z

        lptsky = fieldsky.FieldSky(ID = self.ID+'_'+str(seed),
                                   N  = self.N,
                                 Lbox = self.Lbox,
                                Nside = self.Nside,
                                input = "cube",
                                  gpu = self.gpu,
                                  mpi = self.mpi,
                                 cube = cube)

        lptsky.generate()
        times = xglogutil.profiletime(None, 'field mapping', times, self.comm, self.mpiproc)

        return 0

