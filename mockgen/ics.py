import jax.numpy as jnp
import numpy as np

class ICs:
    '''ICs'''
    def __init__(self, sky, cosmo, cube, format='nyx', fname='testics'):
        self.sky    = sky
        self.cosmo  = cosmo
        self.cube   = cube
        self.format = format
        self.fname  = fname

    def writenyx(self,x,y,z,vx,vy,vz,mass):

    # Nyx format:
    #   fwrite(&npart, sizeof(long), 1, outFile);
    #   fwrite(&DM, sizeof(int), 1, outFile); <-- DM=3 for number of dimensions
    #   fwrite(&NX, sizeof(int), 1, outFile); <-- NX=0 for number of extra fields
    #   for(i=0; i<Npart; i++) {
    #     fwrite(&x[i], sizeof(float), 1, outFile);
    #     fwrite(&y[i], sizeof(float), 1, outFile);
    #     fwrite(&z[i], sizeof(float), 1, outFile);
    #     fwrite(&mass[i], sizeof(float), 1, outFile);
    #     fwrite(&vx[i], sizeof(float), 1, outFile);
    #     fwrite(&vy[i], sizeof(float), 1, outFile);
    #     fwrite(&vz[i], sizeof(float), 1, outFile);
    #   }

    # Nyx units:
    #   position: Mpc
    #   velocity: km/s (peculiar proper velocity)
    #   mass:     M⊙

        fid = open('./output/'+self.fname,'wb')

        npart = self.sky.N**3 / self.sky.nproc
        ndim  = 3
        nx    = 0

        mass = np.repeat(mass, npart)

        np.asarray([npart],dtype=   'long').tofile(fid)
        np.asarray( [ndim],dtype=  'int32').tofile(fid)
        np.asarray(   [nx],dtype=  'int32').tofile(fid)
        np.asarray(      x,dtype='float32').tofile(fid)
        np.asarray(      y,dtype='float32').tofile(fid)
        np.asarray(      z,dtype='float32').tofile(fid)
        np.asarray(   mass,dtype='float32').tofile(fid)
        np.asarray(     vx,dtype='float32').tofile(fid)
        np.asarray(     vy,dtype='float32').tofile(fid)
        np.asarray(     vz,dtype='float32').tofile(fid)

        fid.close()

    def writeics(self):
        cosmo = self.cosmo
        sky  = self.sky

        h      = cosmo.params['h']
        omegam = cosmo.params['Omega_m']

        rho   = 2.775e11 * omegam * h**2
        N     = sky.N
        Lbox  = sky.Lbox
        z     = sky.zInit
        a     = 1 / (1+z)
        Nslab = N // sky.nproc
        j0    = sky.mpiproc * Nslab

        H = 100 * h * jnp.sqrt(omegam*(1+z)**3+1-omegam) # flat universe with negligible radiation

        # LPT position is
        #   x(q) = q + D * S^(1) + b0 * D**2 * S^(2)
        # with
        #   b0 := 3/7 * Omegam_m^(-1/143)
        # and peculiar velocity is
        #   v(q) = a * dx/dt
        #        = a * [ dD/dt * S^(1) + 2 * b0 * D * dD/dt * S^(2) ]
        #        = a * dD/dt * [ S^(1) + 2 * b0 * D * S^(2) ]
        #        = a * f * H * [ S^(1) + 2 * b0 * D * S^(2) ]
        # where
        #   f := dlnD/dlna (= 1 for z>>1) 

        f = 1 # note we are assuming z>>1 here for the ICs!!!

        D  = cosmo.growth_factor_D(z)
        b0 = 3/7 * omegam**(-1/143)

        mass = rho * Lbox**3 / N**3
        if self.sky.mpiproc == 0:
            print(f"particle mass: {mass:.3e} Msun")
            print(f"Lbox:          {Lbox} Mpc")
            print(f"growth factor: {D:.3e}")
            print(f"LPT b0 factor: {b0:.3e}")

        Dgrid = Lbox / N       # grid spacing in Mpc

        x0 = 0.5 * Dgrid        # first point in x/z directions
        x1 = Lbox - 0.5 * Dgrid #  last point in x/z directions

        y0 = (j0 + 0.5) * Dgrid     # first point in y-direction (sharding is hardcoded here!)
        y1 = y0 + (Nslab-1) * Dgrid #  last point in y-direction (sharding is hardcoded here!)

        q1d  = jnp.linspace(x0,x1,N)
        q1dy = jnp.linspace(y0,y1,Nslab)

        qx, qy, qz = jnp.meshgrid(q1d,q1dy,q1d,indexing='ij')

        x =  qx + D * self.cube.s1x + b0 * D**2 * self.cube.s2x
        y =  qy + D * self.cube.s1y + b0 * D**2 * self.cube.s2y
        z =  qz + D * self.cube.s1z + b0 * D**2 * self.cube.s2z

        vx = a * f * H * (self.cube.s1x + 2 * b0 * D * self.cube.s2x)
        vy = a * f * H * (self.cube.s1y + 2 * b0 * D * self.cube.s2y)
        vz = a * f * H * (self.cube.s1z + 2 * b0 * D * self.cube.s2z)

        if self.format == 'nyx':
            self.writenyx(x,y,z,vx,vy,vz,mass)

        return
