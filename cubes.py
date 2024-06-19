import numpy as np
import sys

class Cube:
    ''' Class to read in a cube file, and split up real/imag and alpha/beta
        components. 

        J. Kasper updated and adapted this for Python3, and additional support in
        automatic plotting.
    '''
    def __init__(self, filename,vspin=False):
        self.comment = []
        self.sign    = None # sign on number atoms determines if we are dealing
                            # with total density or a molecular orbital.
        self.natoms  = None
        self.origin  = None 
        self.nx      = None 
        self.ny      = None 
        self.nz      = None 
        self.x       = None 
        self.y       = None 
        self.z       = None 
        self.atoms   = []
        # GHF only; do cubegen with vspin to extract magnetization
        self.vspin   = vspin
        with open(filename,'r') as f:
           # Save comments (first two lines)
           for i in range(2):
               self.comment.append(f.readline().strip('\n')) 
           # Grab number of atoms and the cubefile origin
           line = f.readline().split()
           self.natoms = abs(int(line[0]))
           self.sign   = np.sign(int(line[0]))
           self.origin = np.array([float(line[1]),
                                   float(line[2]),
                                   float(line[3])])
           # Grab number voxels and axis vector
           for vector in [['nx','x'],['ny','y'],['nz','z']]:
               line = f.readline().split()
               self.__dict__[vector[0]] = int(line[0])
               self.__dict__[vector[1]] = np.array([float(line[1]),
                                                    float(line[2]),
                                                    float(line[3])])
           # Grab atoms and coordinates 
           for atom in range(self.natoms):
               line = f.readline().split()
               self.atoms.append([line[0],line[1],line[2],line[3],line[4]])
           # Skip next line ... not sure what it means in new cubegen
           if self.sign < 0:
               f.readline()
           # Grab volumetric data, which finishes out our file
           vals = [float(v) for s in f for v in s.split()]
           # now we need to see what kind of data we have
           # (real/complex?,two-component)

           if(self.vspin):
               if len(vals) == 4*self.nx*self.ny*self.nz:
                   self.N = np.zeros((self.nx,self.ny,self.nz))
                   self.Mx = np.zeros((self.nx,self.ny,self.nz))
                   self.My = np.zeros((self.nx,self.ny,self.nz))
                   self.Mz = np.zeros((self.nx,self.ny,self.nz))
                   for idx,v in enumerate(vals):
                       i = int(np.floor(idx/4))
                       if idx % 4 == 0:
                           self.N[int(i/(self.ny*self.nz)),
                                     int(i/self.nz)%self.ny,
                                      i%self.nz] = float(v)
                       elif idx % 4 == 1:
                           self.Mx[int(i/(self.ny*self.nz)),
                                     int(i/self.nz)%self.ny,
                                      i%self.nz] = float(v)
                       elif idx % 4 == 2:
                           self.My[int(i/(self.ny*self.nz)),
                                     int(i/self.nz)%self.ny,
                                      i%self.nz] = float(v)
                       elif idx % 4 == 3:
                           self.Mz[int(i/(self.ny*self.nz)),
                                     int(i/self.nz)%self.ny,
                                      i%self.nz] = float(v)
               
           elif len(vals) == self.nx*self.ny*self.nz:
               self.volRA = np.zeros((self.nx,self.ny,self.nz))
               for idx,v in enumerate(vals):
                   self.volRA[int(idx/(self.ny*self.nz)),
                           int(idx/self.nz)%self.ny,
                            idx%self.nz] = float(v)
               self.volNorm = np.sqrt(self.volRA**2)
           elif len(vals) == 2*self.nx*self.ny*self.nz:
               # This is the complex one component case
               self.volRA = np.zeros((self.nx,self.ny,self.nz))
               self.volIA = np.zeros((self.nx,self.ny,self.nz))
               for idx,v in enumerate(vals):
                   i = int(np.floor(idx/2))
                   if idx % 2 == 0:
                       self.volRA[int(i/(self.ny*self.nz)),
                                 int(i/self.nz)%self.ny,
                                  i%self.nz] = float(v)
                   elif idx % 2 == 1:
                       self.volIA[int(i/(self.ny*self.nz)),
                                 int(i/self.nz)%self.ny,
                                  i%self.nz] = float(v)
               self.volNorm = np.sqrt(self.volRA**2 +
                                      self.volIA**2 )
           elif len(vals) == 4*self.nx*self.ny*self.nz:
               # This is the complex 2-component case, so we have to split the
               # volumetric data into four parts. The density is stored real
               # alpha, imaginary alpha, real beta, imaginary beta 
               self.volRA = np.zeros((self.nx,self.ny,self.nz))
               self.volIA = np.zeros((self.nx,self.ny,self.nz))
               self.volRB = np.zeros((self.nx,self.ny,self.nz))
               self.volIB = np.zeros((self.nx,self.ny,self.nz))
               for idx,v in enumerate(vals):
                   i = int(np.floor(idx/4))
                   if idx % 4 == 0:
                       self.volRA[int(i/(self.ny*self.nz)),
                                 int(i/self.nz)%self.ny,
                                  i%self.nz] = float(v)
                   elif idx % 4 == 1:
                       self.volIA[int(i/(self.ny*self.nz)),
                                 int(i/self.nz)%self.ny,
                                  i%self.nz] = float(v)
                   elif idx % 4 == 2:
                       self.volRB[int(i/(self.ny*self.nz)),
                                 int(i/self.nz)%self.ny,
                                  i%self.nz] = float(v)
                   elif idx % 4 == 3:
                       self.volIB[int(i/(self.ny*self.nz)),
                                 int(i/self.nz)%self.ny,
                                  i%self.nz] = float(v)
               self.volNorm = np.sqrt(self.volRA**2 +
                                      self.volRB**2 +
                                      self.volIA**2 +
                                      self.volIB**2) 
       
           else:
               raise NameError('Cube file not valid')

    def compute_2c(self):
        self.maga = np.sqrt(self.volRA**2 + self.volIA**2)
        self.arga = np.arctan2(self.volIA,self.volRA)
        self.magb = np.sqrt(self.volRB**2 + self.volIB**2)
        self.argb = np.arctan2(self.volIB,self.volRB)
        self.Mx = 2 * (self.volRA * self.volRB - self.volIA * self.volIB)
        self.My = -2 * (self.volIA * self.volRB - self.volRA * self.volIB)
        self.Mz = self.volRA**2 + self.volIA**2 - self.volRB**2 - self.volIB**2
        self.N = np.sqrt(self.volRA**2 + self.volIA**2 + self.volRB**2 + self.volIB**2)

    def write_xyz(self,filename):
        ''' Creates an xyz file with the coordinates
            given in the cube file.
        '''
        b2a = 0.529177
        with open(filename, 'w') as f:
            print(" %4d " % (self.natoms),file=f)
            print(" ",file=f)
            for atom in self.atoms:
                print( " %s %f %f %f " % (atom[0], float(atom[2])*b2a, 
                      float(atom[3])*b2a, float(atom[4])*b2a),file=f)

    def write_out(self,filename,data='RA'):
        if data == 'RA':
            volume = self.volRA
        elif data == 'IA':
            volume = self.volIA
        elif data == 'RB':
            volume = self.volRB
        elif data == 'IB':
            volume = self.volIB
        elif data == 'NORM':
            volume = self.volNorm 
        
        # Below are some options to print the Magnitude (Mag) and Argument (Arg)
        # of complex cubes, e.g. if z = a + i*b -> |z|*exp(i*theta) where |z| is 
        # the Magnitude (Mag) and theta is the Argument (Arg)
        elif data == 'MagA':
            volume = np.sqrt(self.volRA**2 + self.volIA**2)
        elif data == 'ArgA':
            volume = (np.arctan2(self.volRA,self.volIA))
        elif data == 'MagB':
            volume = np.sqrt(self.volRB**2 + self.volIB**2)
        elif data == 'ArgB':
            volume = (np.arctan2(self.volRB,self.volIB))
        # this the magnitude and angle between alpha and beta components for GHF
        elif data == 'MagAB':
            volume = np.sqrt(self.volRA**2 + self.volIA**2 + self.volRB**2 + self.volIB**2)
        elif data == 'phase':
            volume = np.arctan(self.volIA,self.volRA) + np.arctan(self.volIB,self.volRB)
        elif data == 'ArgAB':
            volume = (abs(np.arctan2(self.volRA,self.volRB)))

        with open(filename, 'w') as f:
            for i in self.comment:
                print(str(i),file=f)
            print(" %4d %.6f %.6f %.6f" % (self.sign*self.natoms, 
                self.origin[0], self.origin[1],self.origin[2]),file=f)
            print(" %4d %.6f %.6f %.6f" % (self.nx, self.x[0], self.x[1],
                self.x[2]), file=f)
            print(" %4d %.6f %.6f %.6f" % (self.ny, self.y[0], self.y[1],
                self.y[2]), file=f)
            print(" %4d %.6f %.6f %.6f" % (self.nz, self.z[0], self.z[1],
                self.z[2]), file=f)
            for atom in self.atoms:
                print(" %s %s %s %s %s" % (atom[0], atom[1], atom[2], atom[3],
                    atom[4]), file=f)
            if self.sign < 0:
                print("    1    "+str(self.natoms),file=f)
            for ix in range(self.nx):
                for iy in range(self.ny):
                    for iz in range(self.nz):
                        print(" %.5e " % volume[ix,iy,iz],end="",file=f)
                        if (iz % 6 == 5):
                            print('',file=f)
                    print('',file=f)
 

