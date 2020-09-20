import numpy as np

class Snapshot():
    def __init__(self):
        print("Instantiated a snapsot object, use from_binary method to read from binary.")
        pass

    def from_binary(self,filename = None,header=True):
        assert type(filename) is str, "This class requires the gadget filename as input"
        self.file = open(filename,'rb')
        if header==True:
            self.read_header()
        else:
            pass
        self.prtcl_types = ["Gas","Halo","Disk",  "Bulge", "Stars", "Bndry"]

    def read_header(self):
        header_size = np.fromfile(self.file, dtype=np.uint32, count=1)
        print ("reading the first block (header) which contains ", header_size, " bytes")
        self.N_prtcl_thisfile  = np.fromfile(self.file, dtype=np.uint32, count=6)    ## The number of particles of each type present in the file
        self.mass_table       = np.fromfile(self.file, dtype=np.float64, count=6)     ## Gives the mass of different particles
        self.scale_factor     = np.fromfile(self.file, dtype=np.float64, count=1)[0]   ##Time of output,  or expansion factor for cosmological simulations
        self.redshift         = np.fromfile(self.file, dtype=np.float64, count=1)[0]   ## REdshift of the snapshot
        self.flag_sfr         = np.fromfile(self.file, dtype=np.int32, count=1)[0]     ##Flag for star 
        self.flag_feedback    = np.fromfile(self.file, dtype=np.int32, count  = 1)[0]  ##Flag for feedback
        self.N_prtcl_total     = np.fromfile(self.file, dtype=np.uint32, count = 6)  ## Total number of each particle present in the simulation
        self.flag_cooling     = np.fromfile(self.file, dtype=np.int32, count =1)[0]     ## Flag used for cooling
        self.num_files        = np.fromfile(self.file, dtype=np.int32, count = 1)[0] ## Number of files in each snapshot
        self.box_size         = np.fromfile(self.file, dtype = np.float64, count = 1)[0]  ## Gives the box size if periodic boundary conditions are used
        self.Omega_m_0        = np.fromfile(self.file, dtype = np.float64, count=1)[0]     ## Matter density at z = 0 in the units of critical density
        self.Omega_Lam_0      = np.fromfile(self.file, dtype = np.float64, count=1)[0]## Vacuum Energy Density at z=0 in the units of critical density
        self.Hubble_param     = np.fromfile(self.file, dtype = np.float64, count =1 )[0] ## gives the hubble constant in units of 100 kms^-1Mpc^-1  
        self.flag_stellar_age = np.fromfile(self.file, dtype = np.int32 , count =1)[0]  ##Creation time of stars
        self.flag_metals      = np.fromfile(self.file, dtype = np.int32 , count =1)[0] ##Flag for metallicity values
        self.N_prtcl_total_HW  = np.fromfile(self.file, dtype = np.int32, count = 6) ## For simulations more that 2^32 particles this field holds the most significant word of the 64 bit total particle number,  otherwise 0
        self.flag_entropy_ICs = np.fromfile(self.file, dtype = np.int32, count = 1)[0] ## Flag that initial conditions contain entropy instead of thermal energy in the u block
        self.file.seek(256 +4 , 0)
        header_size_end = np.fromfile(self.file, dtype = np.int32, count =1)[0]
        print ('Header block is read and it contains ', header_size_end, 'bytes.')
        
        
    def positions(self, prtcl_type="Halo",max_prtcl=None):
        self.file.seek(256+8, 0)
        position_block_size = np.fromfile(self.file, dtype = np.int32, count =1)[0]
        print ("reading the second block (position) which contains ", position_block_size, " bytes")
        i = 0
        while self.prtcl_types[i] != prtcl_type:
            self.file.seek(self.N_prtcl_thisfile[i]*3*4, 1)
            i += 1
        N_prtcl = self.N_prtcl_thisfile[i] if max_prtcl is None else max_prtcl
        posd = np.fromfile(self.file, dtype = np.float32, count = N_prtcl*3)  ### The positions are arranged in the binary file as follow: x1,y1,z1,x2,y2,z2,x3,y3,z3 and so on till xn,yn,zn
        posd = posd.reshape((N_prtcl, 3))   ## reshape keeps the fastest changing axis in the end, since x,y,z dimensions are the ones changing the fastest they are given the last axis.
        if max_prtcl is not None:
            print('Positions of {} particles is read'.format(N_prtcl))
        else:
            end  = np.fromfile(self.file, dtype = np.int32, count =1)[0]
            print ('Position block is read and it contains ', end, 'bytes')
        return posd