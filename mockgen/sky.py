import sys
import os
import logging
import mockgen.defaults as mgd
import xgcosmo.cosmology as xgc
import xgutil.log_util as xglogutil
import xgutil.backend  as xgback
import jax

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
        self.nlpt     = kwargs.get(    'nlpt',mgd.nlpt)
        self.gpu      = kwargs.get(     'gpu',mgd.gpu)
        self.mpi      = kwargs.get(     'mpi',mgd.mpi)
        self.h        = kwargs.get(       'h',mgd.h)
        self.omegam   = kwargs.get(  'omegam',mgd.omegam)
        self.lptw     = kwargs.get(    'lptw',mgd.lptw)

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

        jax.distributed.initialize()

        self.comm    = None
        self.nproc   = jax.device_count()
        self.mpiproc = jax.process_index()
        self.task_tag = "JAX process "+str(self.mpiproc)

        self.parallel = False
        if self.nproc > 1: self.parallel = True

        if self.mpiproc == 0:
            xglogutil.parprint(f'\nSky class initialized running on {self.nproc} GPUs')

    def get_power_array(self,cosmo_wsp):
        import numpy as np
        k = np.logspace(-3,1,1000)
        pk = cosmo_wsp.matter_power(k)
        pk /= self.h**3 # convert power spectrum units from (Mpc/h)^3 to Mpc^3
        k  *= self.h    # convert wavenumber units from h/Mpc to 1/Mpc
        result = np.asarray([k,pk])
        return result

    def run(self, **kwargs):
        import lpt
        from time import time
        times={'t0' : time()}

        if not self.parallel:
            cube = lpt.Cube(N=self.N,Lbox=self.Lbox,nlpt=self.nlpt,partype=None)
        else:
            cube = lpt.Cube(N=self.N,Lbox=self.Lbox,nlpt=self.nlpt)

        if self.laststep == 'init':
            return 0

        backend = xgback.Backend(force_no_gpu=True,force_no_mpi=True,logging_level=-logging.ERROR)
        cosmo_wsp = xgc.cosmology(backend, h=self.h, Omega_m=self.omegam, cosmo_backend='CAMB') # for background expansion consistent with websky

        err = 0
        seeds = range(self.seed,self.seed+self.Niter)
        i = 0
        for seed in seeds:
            if i==1:
                times={'t0' : time()}
            err += self.generatesky(seed,cube,times,cosmo_wsp)
            i += 1
        xglogutil.summarizetime(None,times,self.comm, self.mpiproc)
        
        return err

    def generatesky(self, seed, cube, times, cosmo_wsp,**kwargs):
        from time import time
        import datetime

        import jax
        import jax.numpy as jnp
        import lpt
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
        pofk = self.get_power_array(cosmo_wsp)
        times = xglogutil.profiletime(None, 'get_power_array', times, self.comm, self.mpiproc)
        delta = cube.noise2delta(delta,pofk)
        times = xglogutil.profiletime(None, 'noise convolution', times, self.comm, self.mpiproc)
        if self.laststep == 'convolution':
            return 0

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
                
        #### LPT DISPLACEMENTS FROM DENSITY CONTRAST
        if self.nlpt > 0:
            cube.slpt(infield=self.input,delta=delta) # s1 and s2 in cube.[s1x,s1y,s1z,s2x,s2y,s2z]
            times = xglogutil.profiletime(None, '2LPT', times, self.comm, self.mpiproc)
            if self.lptw:
                fname=f'{self.ID}-{seed}_{self.nlpt}lpt'
                if self.nproc > 1: fname=fname+f'_slab{self.mpiproc}'
                cube.save_displacements(fname)
        if self.laststep == 'lpt':
            return 0

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        from xgfield import fieldsky
        lptsky = fieldsky.FieldSky(ID = self.ID+'_'+str(seed),
                                   N  = self.N,
                                 Lbox = self.Lbox,
                                Nside = self.Nside,
                                 nlpt = self.nlpt,
                                input = "cube",
                                  gpu = self.gpu,
                                  mpi = self.mpi,
                                 cube = cube,
                                 cwsp = cosmo_wsp)

        lptsky.generate()
        times = xglogutil.profiletime(None, 'field mapping', times, self.comm, self.mpiproc)

        return 0

