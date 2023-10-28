import sys
import os
import mockgen.defaults as mgd

def _test_transfer():
    import jax.numpy as jnp
    k  = jnp.logspace(-3,2,1000)
    pk = jnp.sqrt(1e5 * (k/1e-2) * ((1+(k/1e-2)**2)/2)**-4) # something reasonable for testing purposes
    return jnp.asarray([k,pk]).T

class Sky:
    '''Sky'''
    def __init__(self, **kwargs):

        self.ID       = kwargs.get(      'ID',mgd.ID)
        self.N        = kwargs.get(       'N',mgd.N)
        self.seed     = kwargs.get(    'seed',mgd.seed)
        self.input    = kwargs.get(   'input',mgd.input)
        self.Lbox     = kwargs.get(    'Lbox',mgd.Lbox)
        self.laststep = kwargs.get('laststep',mgd.laststep)
        self.Nside    = kwargs.get(   'Nside',mgd.Nside)
        self.gpu      = kwargs.get(     'gpu',mgd.gpu)
        self.mpi      = kwargs.get(     'mpi',mgd.mpi)

    def generate(self, **kwargs):
        from time import time
        import datetime
        times={'t0' : time()}

        import jax
        import lpt
        from xgfield import fieldsky
        from mpi4py import MPI
        import xgutil.log_util as xglogutil
        jax.config.update("jax_enable_x64", True)

        parallel = False
        nproc    = MPI.COMM_WORLD.Get_size()
        mpiproc  = MPI.COMM_WORLD.Get_rank()
        comm     = MPI.COMM_WORLD
        task_tag = "MPI process "+str(mpiproc)

        if MPI.COMM_WORLD.Get_size() > 1: parallel = True

        if mpiproc == 0:
            xglogutil.parprint(f'\nRunning Sky.generate for model "{self.ID}" on {nproc} MPI processes\n')

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
        delta = cube.noise2delta(delta,_test_transfer())
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

        lptsky = fieldsky.FieldSky(ID = self.ID,
                                   N  = self.N,
                                 Lbox = self.Lbox,
                                Nside = self.Nside,
                                input = cube,
                                  gpu = self.gpu,
                                  mpi = self.mpi,
                                 cube = cube)

        lptsky.generate()
        times = xglogutil.profiletime(None, 'field mapping', times, comm, mpiproc)

        return 0

