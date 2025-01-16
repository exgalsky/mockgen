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

    # Nyx format
    # see https://amrex-astro.github.io/Nyx/docs_html/ICs.html#start-from-a-binary-file
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
        nx    = 4

        mass = np.repeat(mass, npart)

        np.asarray([npart],dtype=   'long').tofile(fid)
        np.asarray( [ndim],dtype=  'int32').tofile(fid)
        np.asarray(   [nx],dtype=  'int32').tofile(fid)
    
        (np.asarray([x.flatten(),y.flatten(),z.flatten(),mass,vx.flatten(),vy.flatten(),vz.flatten()], dtype='float32').T).tofile(fid)

        fid.close()

    def write_lc(self,x,y,z):

    # Nyx format
    # see https://amrex-astro.github.io/Nyx/docs_html/ICs.html#start-from-a-binary-file
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
        nx    = 4

        np.asarray([npart],dtype=   'long').tofile(fid)
        np.asarray( [ndim],dtype=  'int32').tofile(fid)
        np.asarray(   [nx],dtype=  'int32').tofile(fid)
    
        (np.asarray([x.flatten(),y.flatten(),z.flatten()], dtype='float32').T).tofile(fid)

        fid.close()

    def writeics(self):
        cosmo = self.cosmo
        sky  = self.sky

        h      = cosmo.params['h']
        omegam = cosmo.params['Omega_m']

        rho   = 2.77536627e11 * omegam * h**2  # rho_c = 2.77536627e11 h^2 Msun/Mpc^3 at z=0
        N     = sky.N
        Lbox  = sky.Lbox
        z_ini = sky.zInit
        a     = 1 / (1+z_ini)
        Nslab = N // sky.nproc
        j0    = sky.mpiproc * Nslab

        H = self.cosmo.Hubble_H(z_ini) #100 * h * jnp.sqrt(omegam*(1+z)**3+1-omegam) # flat universe with negligible radiation

        print("Computing and writing ICs ---->")
        print("Cosmology parameters of interest:")
        print("h = ", cosmo.params['h'])
        print("Omega_m = ", cosmo.params['Omega_m'])
        print("Omega_b = ", cosmo.params['Omega_b'])
        print("YHe = ", cosmo.params['YHe'])
        print("z_init = ", sky.zInit)

        # LPT position is
        #   x(q) = q + D * S^(1) + b0 * D**2 * S^(2)
        # with
        #   b0 := 3/7 * Omegam_m^(-1/143)
        # and peculiar velocity is
        #   v(q) = a * dx/dt
        #        = a * [ dD/dt * S^(1) + 2 * b0 * D * dD/dt * S^(2) ]
        #        = a * dD/dt * [ S^(1) + 2 * b0 * D * S^(2) ]
        #        = a * f * H * [ D *S^(1) + 2 * b0 * D**2* S^(2) ]
        # where
        #   f := dlnD/dlna (= 1 for z>>1) 

        f = 1 # note we are assuming z>>1 here for the ICs!!!

        D  = cosmo.growth_factor_D(z_ini)
        
        print("Growth factor = ", D)
        
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

        Xx =  qx + D * self.cube.s1x + b0 * D**2 * self.cube.s2x
        Xy =  qy + D * self.cube.s1y + b0 * D**2 * self.cube.s2y
        Xz =  qz + D * self.cube.s1z + b0 * D**2 * self.cube.s2z

        vx = a * f * H * (D * self.cube.s1x + 2 * b0 * D**2 * self.cube.s2x)
        vy = a * f * H * (D * self.cube.s1y + 2 * b0 * D**2 * self.cube.s2y)
        vz = a * f * H * (D * self.cube.s1z + 2 * b0 * D**2 * self.cube.s2z)

        if self.format == 'nyx':
            self.writenyx(Xx,Xy,Xz,vx,vy,vz,mass)

        return

    def lightcone(self):
        cosmo  = self.cosmo
        sky    = self.sky
        zmin   = 0.05  # minimum redshift for projection (=0.05 for websky products)
        zmax   = 4.5   # maximum redshift for projection (=4.50 for websky products)

        h      = cosmo.params['h']
        omegam = cosmo.params['Omega_m']

        rho    = 2.775e11 * omegam * h**2
        N      = sky.N
        Lbox   = sky.Lbox
        Nslab  = N // sky.nproc
        j0     = sky.mpiproc * Nslab

        print("Computing and writing lightcones ---->")
        print("Cosmology parameters of interest:")
        print("h = ", cosmo.params['h'])
        print("Omega_m = ", cosmo.params['Omega_m'])
        print("Omega_b = ", cosmo.params['Omega_b'])
        print("YHe = ", cosmo.params['YHe'])
        print("z_min = ", zmin)
        print("z_max = ", zmax)

        mass = rho * Lbox**3 / N**3
        if self.sky.mpiproc == 0:
            print(f"particle mass: {mass:.3e} Msun")
            print(f"Lbox:          {Lbox} Mpc")

        Dgrid = Lbox / N       # grid spacing in Mpc

        # Setting up axes grid points
        x0 = 0.5 * Dgrid        # first point in x/z directions
        x1 = Lbox - 0.5 * Dgrid #  last point in x/z directions

        y0 = (j0 + 0.5) * Dgrid     # first point in y-direction (sharding is hardcoded here!)
        y1 = y0 + (Nslab-1) * Dgrid #  last point in y-direction (sharding is hardcoded here!)

        q1d  = jnp.linspace(x0,x1,N)
        q1dy = jnp.linspace(y0,y1,Nslab)

        # create Lagrange grid
        qx, qy, qz = jnp.meshgrid(q1d,q1dy,q1d,indexing='ij')

        qx = qx.ravel()
        qy = qy.ravel()
        qz = qz.ravel()

        # Get comoving distance and reshift from the comoving distance information
        # Finally get growth fractors from the redshift information
        def comoving_q(x_i, y_i, z_i):
            return jnp.sqrt(x_i**2. + y_i**2. + z_i**2.)

        chi = comoving_q(qx, qy, qz)
        z   = cosmo.comoving_distance2z(chi)
        D   = cosmo.growth_factor_D(z)
        b0 = 3/7 * omegam**(-1/143)

        # Compute Euclidean positions of the particles
        Xx =  qx + D * self.cube.s1x.ravel() + b0 * D**2 * self.cube.s2x.ravel()
        Xy =  qy + D * self.cube.s1y.ravel() + b0 * D**2 * self.cube.s2y.ravel()
        Xz =  qz + D * self.cube.s1z.ravel() + b0 * D**2 * self.cube.s2z.ravel()

        # write the xyz coordinates of the particles to file
        if self.format == 'nyx':
            self.write_lc(Xx,Xy,Xz)

        return 
        