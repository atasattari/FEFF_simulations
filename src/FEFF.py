import numpy as np
from scipy import constants as _constants
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from pathlib import Path
from itertools import product,repeat
from multiprocessing import Pool
from collections.abc import Iterable
import subprocess
from functools import partial
import numba as nb
import argparse
from collections import defaultdict
import glob

class constants:
    '''Physical constants used in the FEFF calculations.
    These constants are used to convert between atomic units and SI units, as well as to define the physical properties of electrons and photons.'''    
    unit_ev_in_joule = _constants.physical_constants['electron volt'][0]
    atomic_unit_of_momentum = _constants.physical_constants['atomic unit of momentum'][0]
    electron_mass = _constants.physical_constants['electron mass energy equivalent in MeV'][0]*1e6
    c = _constants.c
    r0 = _constants.physical_constants['classical electron radius'][0]
    
    momentum_au_to_SI = atomic_unit_of_momentum/unit_ev_in_joule*c # momentum_eV*c, it should be named au to natural units. :/
    energy_au_to_SI = _constants.physical_constants['atomic unit of energy'][0]/unit_ev_in_joule
    
class kinetamic:
    '''Kinematic relations for the FEFF calculations.
    This class provides methods to calculate the transferred momentum, scattering angle, and energy transfer based on the kinematic relations used in X-ray scattering experiments.'''
    @staticmethod
    def get_q(theta, E1, E):
        '''transferred momentum'''
        n1 = 2*E1
        n2 = 1-np.cos(theta)
        n3 = E1-E
        n4 = E*E
        q2c2 = (n1*n2*n3+n4)*constants.unit_ev_in_joule**2
        q2 = q2c2/constants.c**2 # momentum^2 in SI
        atomic_q2 = q2/constants.atomic_unit_of_momentum**2 # a.u^2.
        return np.sqrt(atomic_q2) # a.u.
    
    @staticmethod
    def get_theta(q,E1,E):
        '''scattering angle'''
        q = q*constants.momentum_au_to_SI
        n1 = q**2-E**2
        d1 = 2*E1*(E1-E)
        theta = np.arccos(1-n1/d1)
        return theta # radians
    
    @staticmethod
    def q_approximation(theta, E1):
        '''Aproximate transferred transferred momentum'''
        n1 = np.sin(theta/2)
        q = 2*E1*n1/constants.c*constants.unit_ev_in_joule
        atomic_q = q/constants.atomic_unit_of_momentum
        return atomic_q # a.u.
    
    @staticmethod
    def theta_approximation(q, E1):
        '''approximate scattering angle'''
        theta = 2*np.arcsin(q/2/E1)
        return theta # radians
    
    @staticmethod
    def get_E(q,pq):
        E = q**2/(2*constants.electron_mass)+q*pq/constants.electron_mass #momentum should be in eV*c (e = pc)
        return E #[eV]
    
    @staticmethod
    def get_pq(E, q):
        '''
        E is the energy transfer in eV
        q is the momentum transfer in a.u.
        '''
        E = E/constants.energy_au_to_SI
        pq = E/q - q/2
        return pq # in a.u.

class run_FEFF:
    """Run FEFF calculations for EXAFS and XANES simulations.
    This class handles the configuration, execution, and management of FEFF simulations for different modes (EXAFS, XANES) and edges (L1, L2, L3).
    It generates the necessary input files, runs FEFF in parallel, and manages the output files.
    Parameters:
    config (simulation_config): Configuration object containing simulation parameters.
    FEFF_dir (str): Path to the FEFF executable directory.
    n_cores (int): Number of CPU cores to use for parallel processing.
    verbose (bool): If True, prints detailed information during execution.
    crystal_data (str): Path to the crystal data file.
    
    Attributes:
    crystal (str): Crystal structure data loaded from the specified file.
    compton_path (str): Path for the Compton profile output.
    config (simulation_config): Configuration object containing simulation parameters.
    unique_paths (iterable): Unique paths for each configuration.
    grouped_paths (dict): Grouped paths for configurations by mode and edge.
    all_paths (iterable): All paths for configurations.

    Methods:
    _load_crystal_data(filename): Load crystal data from a text file.
    feff_command(momentum, edge, mode, control=('0 0 1 1 1 1')): Generate FEFF input command for a given configuration.
    feff_compton_command(control=('0 0 1 1 1 1')): Generate FEFF input command for Compton profile.
    parse_config(config): Parse the configuration object to set up paths and modes.
    _generate_directory(path): Create a directory if it does not exist.
    _write_FEFF_inp(command, path): Write the FEFF input command to a file in the specified path.
    write_FEFF_inp(config_iter, control=('0 1 1 1 1 1')): Write FEFF input files for each configuration in the iterable.
    job(path, feff_dir, erase=True): Run FEFF in the specified path and handle output files.
    copy_file(to_move, path): Copy a file to the specified path.
    copy_files(from_path, to_repos): Copy files from one path to multiple destination paths.
    copy_files_in_parallel(): Copy files in parallel
    run_jobs(): Execute the FEFF jobs for unique configurations, copy necessary files, and run all configurations.
    """
    def __init__(self, config ,FEFF_dir=None, n_cores=5,verbose=True,crystal_data='crystal.txt'):
        self.crystal = self._load_crystal_data(crystal_data)
        self.compton_path = None
        self.config = self.parse_config(config)
        self.FEFF_dir = FEFF_dir 
        self.n_cores = n_cores
        self.verbose = verbose
        self.run_jobs()
    
    def _load_crystal_data(self, filename):
        with open(filename, 'r') as file:
            return file.read()

    def feff_command(self, momentum,edge,mode,control=('0 0 1 1 1 1')):
        if mode =="EXAFS":
            info = '''RPATH 5.0\nEXAFS 20'''
        if mode =="XANES":
            info = '''FMS 8.0 \nXANES 6'''
        command = f"""S02 0.0
SCF 7.0 0 30 0.2 3
REAL
CONTROL {control}
PRINT 0 0 0 0 0 0
COREHOLE RPA
*COREHOLE none
EDGE {edge}
{info}
NRIXS -1 {momentum}
LJMAX 15
POTENTIALS
  * ipot   Z      tag
     0     14     Si
     1     14     Si
{self.crystal}"""
        return command
    def feff_compton_command(self,control=('0 0 1 1 1 1')):
        command = f"""S02 0.0
CONTROL {control}
LDOS -30.0 30.0 0.1
PRINT 0 0 0 0 0 0
COREHOLE RPA
SCF 7.0 0 100 0.2 3 
CGRID 10 40 40 40 120
FMS 8. 0     
COMPTON 10  
POTENTIALS
  * ipot   Z      tag
     0     14     Si
     1     14     Si
{self.crystal}"""
        return command
    def parse_config(self, config):
        self.compton_path = config.compton_path
        
        #unique paths
        unique_configs = product(config.modes,config.edges,config.momentums[0:1])
        self.unique_paths = ((mode, edge, momentum0, config._get_path(mode, edge,momentum0)) for mode, edge, momentum0 in unique_configs)
        
        #grouped by mode edge:
        def make_generator(mode,edge, momentums):
            return (config._get_path(mode, edge,momentum) for momentum in momentums)
        grouped_configs = product(config.modes,config.edges)
        self.grouped_paths = {(mode, edge):make_generator(mode,edge, config.momentums) for mode, edge in grouped_configs}

        #all paths
        all_configs = product(config.modes,config.edges,config.momentums)
        self.all_paths = ((mode, edge, momentum, config._get_path(mode, edge,momentum)) for mode, edge, momentum in all_configs)

    def _generate_directory(self,path):      
        path = Path(path)
        path.mkdir(parents=True,exist_ok=True)
    
    def _write_FEFF_inp(self,command,path):      
        self._generate_directory(path)
        with open(f'{path}/feff.inp','w') as f:
            f.write(command)
    
    def write_FEFF_inp(self,config_iter,control=('0 1 1 1 1 1')):
        paths = []
        for mode, edge, momentum, path in config_iter:
            command = self.feff_command(momentum,edge,mode, control)
            self._write_FEFF_inp(command,path)
            paths.append(path)
        if self.compton_path is not None:
            command = self.feff_compton_command(control)
            self._write_FEFF_inp(command,self.compton_path)
            paths.insert(0,self.compton_path)
        return paths

    @staticmethod
    def job(path,feff_dir,erase=True):
        log = open(f"{path}/feff_log.txt", "w")
        os.chdir(path)
        subprocess.run(feff_dir,stdout=log)
        
        if erase:
            for clean_up in glob.glob(f'{path}/*.*'):
                if not clean_up.endswith(('xmu.dat','compton.dat')):   # 'feff.inp', 'feff_log.txt'
                    os.remove(clean_up)
        log.write('\nfinished\n')
        log.close()     
    
    @staticmethod
    def copy_file(to_move, path):
        subprocess.call(f'cp {to_move} {path}', shell=True)
    
    def copy_files(self, from_path, to_repos):
        files = glob.glob(f'{from_path}/*.*')
        files.remove(f"{from_path}/feff_log.txt")
        files.remove(f"{from_path}/feff.inp")

        with Pool(processes=self.n_cores) as pool:
                    pool.starmap(self.copy_file, [(to_move, path) for path in to_repos for to_move in files])

    def copy_files_in_parallel(self):
        for group in self.grouped_paths:
            dir_iter = iter(self.grouped_paths[group])
            dir0 = next(dir_iter)
            self.copy_files(dir0, dir_iter)

    def run_jobs(self):
        print('~~~~~~~~Making first files~~~~~~~~')
        unique_paths = self.write_FEFF_inp(self.unique_paths,control=('1 0 0 0 0 0'))    
        if self.verbose:
            return
        print('~~~~~~~~Run feff for unique configs~~~~~~~~')
        f = partial(self.job, feff_dir = self.FEFF_dir,erase=False)
        with Pool(processes=self.n_cores) as pool:
            pool.map(f, unique_paths)
        print('~~~~~~~~Making all files~~~~~~~~')
        all_paths = self.write_FEFF_inp(self.all_paths,control=('0 1 1 1 1 1'))
        print('~~~~~~~~Copying Pots and xphs~~~~~~~~')
        self.copy_files_in_parallel()
        print('~~~~~~~~Running all~~~~~~~~')
        f = partial(self.job, feff_dir = self.FEFF_dir, erase=True)
        with Pool(processes=self.n_cores) as pool:
            pool.map(f, all_paths)

class simulation_config:
    """Configuration for FEFF simulations.
    This class handles the modes, edges, and momentums for the FEFF simulations.
    Parameters:
    modes (list): List of modes for the simulation (e.g., ['EXAFS', 'XANES']).
    edges (list): List of edges for the simulation (e.g., ['L1', 'L2', 'L3']).
    momentums (list or np.array): List or array of momentum values for the simulation.
    top_dir (str): Top directory for the simulation files.
    compton (bool): If True, includes the Compton profile in the simulation.
    Attributes:
    modes (list): List of modes for the simulation.
    edges (list): List of edges for the simulation.
    momentums (np.array): Array of momentum values for the simulation.
    top_dir (str): Top directory for the simulation files.
    compton_path (str): Path for the Compton profile output if compton is True.
    Methods:
    parse_input(modes, edges, momentums): Parse and validate the input parameters.
    _get_path(mode, edge, momentum): Generate the path for a given mode, edge, and momentum.
    __add__(config2): Combine two simulation configurations.
    __iter__(): Iterate over the configurations, yielding mode, edge, momentum, and path.
    """
    def __init__(self, modes, edges, momentums,top_dir,compton=True):
        self.modes, self.edges, self.momentums =  self.parse_input(modes, edges, momentums)
        self.top_dir = top_dir
        self.compton_path = None
        if compton:
            self.compton_path = top_dir +'/compton'

    def parse_input(self, modes, edges, momentums):
        if not isinstance(edges,Iterable):
            edges = [edges]
        if not isinstance(modes,Iterable):
            modes = [modes]
        assert all(mode in ['EXAFS', 'XANES'] for mode in modes)
        if not isinstance(momentums,Iterable):
            momentums = np.array([momentums])
        momentums = np.round(momentums,2)
        return modes, edges, momentums 
    
    def _get_path(self,mode, edge, momentum):
        return f'{self.top_dir}/{mode}/{edge}/momentum_{momentum}'
    
    def __add__(self, config2):
        assert(config2.compton_path == self.compton_path)
        self.modes = np.union1d(self.modes,config2.modes)
        self.edges = np.union1d(self.edges,config2.edges)
        self.momentums = np.union1d(self.momentums,config2.momentums)
        return self
    
    def __iter__(self):
        configs = product(self.modes, self.edges, self.momentums)
        for mode, edge, momentum in configs:
            yield mode, edge, momentum, self._get_path(mode, edge, momentum)

class build_xsection:
    """Build X-ray scattering cross-section profiles.
    This class constructs the X-ray scattering cross-section profiles based on the Compton profile and core-level data.
    Parameters:
    config (simulation_config): Configuration object containing simulation parameters.
    Attributes:
    compton_path (str): Path for the Compton profile output.
    constructed_cp (bool): Flag indicating whether the Compton profile has been constructed.
    config (simulation_config): Configuration object containing simulation parameters.
    electron_count (dict): Dictionary mapping edges to the number of electrons in the core levels.
    Methods:
    parse_config(config): Parse the configuration object to set up paths and modes.
    read_file(path, columns=(0, 3)): Read data from a file and return specified columns.
    normalize_cp(pq, j): Normalize the Compton profile based on the f-sum rule.
    Bethe_f_sum(s, w, q): Calculate the Bethe f-sum for a given energy range and momentum.
    _cache_cp(pq, j): Cache the Compton profile data.
    make_cp_profile(path, columns=(0, 1)): Construct the Compton profile from the data file.
    interp_nb(x_vals, x, y): Interpolate the Compton profile data outside the 'pq' region.
    get_s_compton(momentum_transfer, compton_path='/media/ata/HDD/PhD_data/feff_core_hole/compton/compton.dat'): Get the Compton profile for a given momentum transfer.
    get_s_core_new(XANES_path, EXAFS_path, n_electrons, columns=(0, 3), switch_energy=45): Get the core-level scattering profile by merging XANES and EXAFS data.
    get_s_total_new(edge, valence_energy=(50, 2000, 1)): Get the total scattering profile for a given edge.
    get_2d_integrand_new(E1, edge, valence_energy=(50, 2000, 1)): Get the 2D integrand for the scattering profile.
    get_1d_integrand_new(E1, edge, valence_energy=(50, 2000, 1)): Get the 1D integrand for the scattering profile.
    """
    def __init__(self, config):
        self.compton_path = None
        self.constructed_cp = False
        self.config = self.parse_config(config)
        self.electron_count = dict(L3=4, L2=2, L1=2)
    
        
    def parse_config(self, config):
        if config.compton_path is not None:
            self.compton_path = config.compton_path + '/compton.dat'
        config.get_path = lambda mode, edge, momentum : config._get_path(mode, edge, momentum) + '/xmu.dat'
        return config
    
    def read_file(self, path,columns = (0,3)):
        with open(f'{path}','r') as f:
            output = f.read()
        data = output.split('#')[-1].split('\n')[1:-1]
        column_data = {col: [] for col in columns}
        for line in data:
            parts = line.split()
            for i in columns:
                column_data[i].append(parts[i])
        column_data = {col: np.array(data, dtype=float) for col, data in column_data.items()}
        return column_data

    def normalize_cp(self,pq,j):
        '''
        coming from f-sum rule. PhysRevB.94.214201 --> eqn.(11)
        '''
        dj = (j[1:]-j[:-1])
        dpq = pq[1:] - pq[:-1]
        p = (pq[1:]+pq[:-1])
        ro_p = -1/(np.pi*p)*dj/dpq
        p2 = p*p
        dp = np.diff(p,append=2*p[-1]-p[-2])
        N_sum = 2*np.pi*np.sum(dp*p2*ro_p)
        coeff = 32/N_sum # 32 valence electrons in silicon.
        return coeff 

    @staticmethod
    def Bethe_f_sum(s,w,q):
        '''
        s: callable
        w: energy range [eV]
        q: momentum a.u.

        '''
        dw = np.diff(w,append =2*w[-1]-w[-2])
        a1 = np.sum(s(w)*w*dw)/constants.energy_au_to_SI
        a2 = 2/(q*q)
        return np.round(a1*a2,2)

    def _cache_cp(self,pq,j):
        self.pq, self.j = pq, j 
        
    def make_cp_profile(self,path,columns =(0,1)):
        if not self.constructed_cp:
            data = self.read_file(path,columns)
            pq = data[columns[0]]
            j = data[columns[1]]
            pq = np.concatenate([-pq[:0:-1],pq])
            j = np.concatenate([j[:0:-1],j])
            coeff = self.normalize_cp(pq,j)
            self._cache_cp(pq, j*coeff)
            self.constructed_cp = True
        return self.pq, self.j #pq and j are in atomic

    @staticmethod
    # @nb.njit
    def interp_nb(x_vals, x, y):
        '''
        Interpolation out of the the 'pq' region set to 0. 
        
        Private communication with JJ Rher:        
        I took the time to check Egor's calculation with the COMPTON card  in FEFF9 and a feff.inp for  a small
        71 atom test cluster of Si.  That ran just fine on my laptop and yielded a CP similar to Egor's J(p_q) in Fig. 8.  With that
        data,  a plot of S(q,w) vs q for each value of w can be made by plotting J(p_q)/q  vs q for fixed values of
        w with p_q = w/q - q/2 > 0 and J(p_q)  set to zero outside the data range  0 < p_q <  5,
        With this cutoff, there is no divergence in the limit q-> 0 where p_q > 5 for any w. The data can then be replotted vs
        w for each q by searching for the crossing points. Of course, this relation between S(q,w)  and J(p_q)/q
        assumes the  IA which is only valid for sufficiently large energy transfers.   My compton.dat is attached.
        It wasn't clear from your plots whether you also did such a calculation.  It would be interesting to compare with the
        sum of  NRIXS S_n (q,w) summed  over the semi-core and valence levels as a check on the IA. Let me know if I've missed something.
        '''
        return np.interp(x_vals, x, y,left=0.,right=0.)


    def get_s_compton(self,momentum_transfer,compton_path='/media/ata/HDD/PhD_data/feff_core_hole/compton/compton.dat'):
        pq, j = self.make_cp_profile(compton_path)        
        pq_eV = pq*constants.momentum_au_to_SI
        momentum_transfer_eV = momentum_transfer*constants.momentum_au_to_SI
        E = kinetamic.get_E(momentum_transfer_eV,pq_eV)  
        s = j/momentum_transfer/constants.energy_au_to_SI  #Change units of s from 1/a.u. ->  1/eV 
        mask = E>0
        signal = lambda x_vals: self.interp_nb(x_vals, E[mask],s[mask])
        return signal

    def get_s_core_new(self,XANES_path,EXAFS_path , n_electrons, columns = (0,3),switch_energy = 45):
        xanes_data = self.read_file(XANES_path,columns=columns)
        exafs_data = self.read_file(EXAFS_path,columns=columns)

        E_xanes,S_xanes = xanes_data[columns[0]],xanes_data[columns[1]]
        E_exafs,S_exafs = exafs_data[columns[0]],exafs_data[columns[1]]

        xanes_exafs_switch = E_xanes[0] + switch_energy # Use xanes between ~(-10 to 30-50)eV around the edge, afterward switch to exafs (FEFF document). 
        mask_xanes = E_xanes<xanes_exafs_switch
        E_xanes,S_xanes = E_xanes[mask_xanes],S_xanes[mask_xanes]
        
        mask_exafs = E_exafs>xanes_exafs_switch
        E_exafs, S_exafs = E_exafs[mask_exafs], S_exafs[mask_exafs]
        
        merged_E = np.concatenate([E_xanes,E_exafs])
        merged_S = np.concatenate([S_xanes,S_exafs])*n_electrons
        
        return merged_E, merged_S

    def get_s_total_new(self,edge,valence_energy = (50,2000,1)):
        assert(edge=='valence' or 
               edge=='L1' or
               edge=='L2' or
               edge=='L3')
        if edge != 'valence':
            for momentum in self.config.momentums:
                xanes_adrress = self.config.get_path('XANES',edge,momentum)
                exafs_adrress = self.config.get_path('EXAFS',edge,momentum)
                merged_E, merged_S =  self.get_s_core_new(XANES_path = xanes_adrress, 
                                                        EXAFS_path = exafs_adrress, 
                                                        n_electrons= self.electron_count[edge])
                yield merged_E, repeat(momentum), merged_S
        else:
            valence_callable = lambda momentum : self.get_s_compton(momentum,self.compton_path) 
            merged_E = np.round(np.arange(*valence_energy),1)
            for momentum in self.config.momentums:
                s_valence = valence_callable(momentum)
                yield merged_E, repeat(momentum), s_valence(merged_E)

    def get_2d_integrand_new(self,E1,edge, valence_energy = (50,2000,1)):
        for data in self.get_s_total_new(edge, valence_energy = valence_energy):
            merged_E, momentum, merged_S = data
            q = next(momentum)
            E2 = E1-merged_E
            q_na = q*constants.momentum_au_to_SI
            cos_theta = (E1**2+E2**2-q_na**2)/(2*E1*E2)
            cos2_theta = cos_theta**2
            n1 = (1+cos2_theta)/2
            intergrand = 1/E1**2*n1*merged_S*q
            yield merged_E, momentum, intergrand

    def get_1d_integrand_new(self,E1,edge, valence_energy = (50,2000,1)):
        data_iter = self.get_2d_integrand_new(E1,edge, valence_energy = valence_energy)
        first_merged_E, _, integrand_1d = next(data_iter)
        for data in data_iter:
            merged_E, _, integrand_2d = data
            assert((first_merged_E==merged_E).all())
            integrand_1d+=integrand_2d

        return first_merged_E,integrand_1d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--ncores', type=int, help='number of cores')
    args = parser.parse_args()
    ncores = args.ncores
    modes = ["EXAFS", 'XANES']
    edges = ['L1','L2','L3']
    momentum = np.round(np.arange(0.1,20,0.02),2)
    top_dir = '/home/sattari2/scratch/feff_core_hole'
    config = simulation_config(modes,edges,momentum,top_dir,compton=False)
    FEFF_dir = '/home/sattari2/feff10-10.0.0/bin/feff'
    run_FEFF(config,FEFF_dir,ncores,verbose=False)

def f(theta,E1):
    theta_radians = np.pi/180*theta
    n1 = FEFF.constants.r0**2
    n2 = (1  + np.cos(theta_radians)**2)/2
    E = np.linspace(0,500,500)
    n3 = 1 - E/E1
    s = np.array([graph.Interpolate(energy,theta) for energy in E])
    return n1*n2*n3*s


if __name__ == '__main__':
    main()

