#!/bin/sh
"exec" "$FIDASIM_DIR/deps/python" "$0" "$@"

# prepare input file for MAST FIDASIM calculation


import argparse
import numpy as np
from numpy import linalg as LA
from scipy.interpolate import interp1d
import fidasim as fs
import pickle as PCL


#%% read a pickled dictionary

def read_pcl(filename):
    f = None
    with open(filename, 'rb') as f:
        d = PCL.load(f)
        f.close()
    return d
        



#%%
def setup_cfpd(dfile):
    # setup the cfpd geometry fir FIDASIM from the output of calc_orbits_fidasim.py
    print("setup cfpd data")
    shapes = {'rect':1, 'round':2}
    FD = read_pcl(dfile)
    FD['data_source'] = 'calc_orbits_fidasim.py'

    FD['radius'] = np.array([LA.norm(v) for v in FD['a_cent'].T])
    FD['a_shape'] = np.array([shapes[s] for s in  FD['a_shape']])
    FD['d_shape'] = np.array([shapes[s] for s in  FD['d_shape']])
    FD['id'] = np.array(FD['id']).astype(np.bytes_)
    
    cfpd_keys = ['nchan', 'system', 'data_source', 'id', 'a_shape', 'd_shape', 'a_cent', 'a_redge', 'a_tedge', 'd_cent', 'd_redge', 'd_tedge', 'radius']
    cfpd_geometry = {}
    for k in cfpd_keys:
        cfpd_geometry[k] = FD[k]
    
    cfpd_table = {'sightline':FD['sightline'], 'daomega':FD['daomega'], 'nactual':FD['nactual'], 'nrays':FD['nrays'],
             'nsteps':FD['nsteps'], 'nenergy':FD['nenergy'], 'earray':FD['earray']}
    
    # complete the table
    cfpd_geometry.update(cfpd_table)

    return cfpd_geometry
    
#%% setup neutralbeam geometry

# at this point this could be anything


def setup_beam(beta=0.0):
    
    uvw_src = np.array([0.0, -530.0 - 2.0*np.cos(beta), 2.0*np.sin(beta)])
    uvw_pos = np.array([0.0, -530.0, 0.0])
    uvw_axis = uvw_pos - uvw_src
    uvw_axis = uvw_axis/np.linalg.norm(uvw_axis)

    focy = 999999.9e0
    focz = 1000e0

    divy = np.full(3,8.73e-3)
    divz = np.full(3,2.27e-2)

    widy = 6.0
    widz = 24.0

    naperture = 1
    ashape = np.array([1])
    awidy = np.array([8.85])
    awidz = np.array([24.0])
    aoffy = np.array([0.0])
    aoffz = np.array([0.0])
    adist = np.array([186.1])

    nbi = {"name":"test_beam","shape":1,"data_source":'run_tests:test_beam',
           "src":uvw_src, "axis":uvw_axis, "widy":widy, "widz":widz,
           "divy":divy, "divz":divz, "focy":focy, "focz":focz,
           "naperture":naperture, "ashape":ashape, "adist":adist,
           "awidy":awidy, "awidz":awidz, "aoffy":aoffy, "aoffz":aoffz}

    return nbi


#%% setup viewing cords for FIDA
def setup_chords():
    ulens = np.zeros(3)
    vlens = np.full(3,-170.0)
    wlens = np.full(3,100.0)
    lens = np.vstack((ulens,vlens,wlens))

    ulos = np.zeros(3)
    vlos = np.array([-200.0,-170.0,-140.0])
    wlos = np.zeros(3)
    los = np.vstack((ulos,vlos,wlos))
    axis = los - lens
    for i in range(3):
        axis[:,i] = axis[:,i]/np.linalg.norm(axis[:,i])

    sigma_pi = np.ones(3)
    spot_size = np.zeros(3)
    radius = np.sqrt(ulos**2 + vlos**2)
    id = np.array([b"f1",b"f2",b"f3"])

    chords = {"nchan":3, "system":"SPECTRAL","id":id, "data_source":"run_tests:test_chords",
              "lens":lens, "axis":axis, "spot_size":spot_size, "sigma_pi":sigma_pi,
              "radius":radius}

    return chords

#%% setup NPA vires
def setup_npa():
    nchan = 3
    ulens = np.zeros(nchan)
    vlens = np.repeat(-170.0,nchan)
    wlens = np.repeat(100.0,nchan)
    lens = np.vstack((ulens,vlens,wlens))

    ulos = np.zeros(nchan)
    vlos = np.array([-200.0,-170.0,-140.0])
    wlos = np.zeros(nchan)
    radius = np.sqrt(ulos**2 + vlos**2)
    id = np.array([b"c1",b"c2",b"c3"])

    a_cent  = np.zeros((3,nchan))
    a_redge = np.zeros((3,nchan))
    a_tedge = np.zeros((3,nchan))
    d_cent  = np.zeros((3,nchan))
    d_redge = np.zeros((3,nchan))
    d_tedge = np.zeros((3,nchan))

    ac = np.array([0.0, 0.0, 0.0])
    ar = np.array([0.0, 3.0, 0.0])
    at = np.array([0.0, 0.0, 3.0])

    dc = np.array([-50.0, 0.0, 0.0])
    dr = np.array([-50.0, 3.0, 0.0])
    dt = np.array([-50.0, 0.0, 3.0])

    for i in range(nchan):
        r0 = np.array([ulens[i],vlens[i],wlens[i]])
        rf = np.array([ulos[i],vlos[i],wlos[i]])
        R = fs.utils.line_basis(r0,rf-r0)
        a_cent[:,i]  = np.dot(R, ac) + r0
        a_redge[:,i] = np.dot(R, ar) + r0
        a_tedge[:,i] = np.dot(R, at) + r0

        d_cent[:,i]  = np.dot(R, dc) + r0
        d_redge[:,i] = np.dot(R, dr) + r0
        d_tedge[:,i] = np.dot(R, dt) + r0

    npa_chords = {"nchan":nchan, "system":"NPA", "data_source":"run_tests.py:test_npa",
                  "id":id, "a_shape":np.repeat(2,nchan), "d_shape":np.repeat(2,nchan),
                  "a_cent":a_cent, "a_redge":a_redge, "a_tedge":a_tedge,
                  "d_cent":d_cent, "d_redge":d_redge, "d_tedge":d_tedge,"radius":radius}

    return npa_chords



#%% main setup code creating input files
def run_main_setup(args):
    fida_dir = fs.utils.get_fidasim_dir()
    test_dir = fida_dir + '/MAST'

    einj = 59.585 #keV
    pinj = 1.62044   #MW

    cgfitf = np.array([-0.109171,0.0144685,-7.83224e-5])
    cgfith = np.array([0.0841037,0.00255160,-7.42683e-8])
    ffracs = cgfitf[0] + cgfitf[1]*einj + cgfitf[2]*einj**2
    hfracs = cgfith[0] + cgfith[1]*einj + cgfith[2]*einj**2
    tfracs = 1.0-ffracs-hfracs
    current_fractions = np.array([ffracs,hfracs,tfracs])

    basic_inputs = {"device":"MAST", 
                    "shot":29904, 
                    "time":0.214,
                    "einj":einj, 
                    "pinj":pinj, 
                    "current_fractions":current_fractions,
                    "ab":2.01410178e0, 
                    "lambdamin":647e0, 
                    "lambdamax":667e0, 
                    "nlambda":2000,
                    "n_fida":50000, 
                    "n_npa":50000, 
                    "n_nbi":500,
                    "n_pfida":500000, 
                    "n_pnpa":500000,
                    "n_halo":5000, 
                    "n_dcx":5000, 
                    "n_birth":100,
                    "ne_wght":50, 
                    "np_wght":50,
                    "nphi_wght":100,
                    "emax_wght":100e0,
                    "nlambda_wght":1000,
                    "lambdamin_wght":647e0,
                    "lambdamax_wght":667e0,
                    "calc_npa":0, 
                    "calc_brems":0,
                    "calc_fida":0,
                    "calc_neutron":0,
                    "calc_cfpd":1, 
                    "calc_res":0,
                    "calc_bes":0, 
                    "calc_dcx":0, 
                    "calc_halo":0, 
                    "calc_cold":0,
                    "calc_birth":0, 
                    "calc_fida_wght":0,
                    "calc_npa_wght":0,
                    "calc_pfida":0, 
                    "calc_pnpa":0,
                    "result_dir":args.path, 
                    "tables_file":fida_dir+'/tables/atomic_tables.h5'}

    basic_bgrid = {"nx":50, 
                   "ny":60, 
                   "nz":70,
                   "xmin":-200.0, 
                   "xmax":200.0,
                   "ymin":-200.0,
                   "ymax":200.0,
                   "zmin":-200.0, 
                   "zmax":200.0,
                   "alpha":0.0, 
                   "beta":0.0, 
                   "gamma":0.0,
                   "origin":np.zeros(3)}

    inputs = basic_inputs.copy()
    
    inputs.update(basic_bgrid)
    
    inputs["comment"] = "Non-rotated, Non-tilted grid, realistic profiles"
    inputs["runid"] = args.runid

    nbi = setup_beam()
    spec = setup_chords()
    npa = setup_npa()
    cfpd_geo = setup_cfpd('Test_detector_array1.pcl')

    # directories of TRANSP output
    TRANSP_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/MAST_data/TRANSP/29904/'
    EQDSK_DIR = '/Users/boeglinw/Documents/boeglin.1/Fusion/Fusion_Products/orbit_mod90_calc/FIDASIM/data/'

    #setup grid for R-Z
    grid = fs.utils.rz_grid(0.01, 200.0, 70, -200.0, 200.0, 100)

    equil, rhogrid, btipsign = fs.utils.read_geqdsk(EQDSK_DIR+'g029906.00214.dat', grid, ccw_phi=True, exp_Bp=0)
    fbm = fs.utils.read_nubeam(TRANSP_DIR + '29904O04_fi_1.cdf', grid, btipsign = btipsign)
    
    plasma = fs.utils.extract_transp_plasma(TRANSP_DIR +'29904O04.CDF', basic_inputs['time'], grid, rhogrid)
    plasma['deni'] = plasma['deni'] - fbm['denf'].reshape(1,grid['nr'],grid['nz'])
    plasma['deni'] = np.where(plasma['deni'] > 0.0, plasma['deni'], 0.0).astype('float64')

    fs.prefida(inputs, grid, nbi, plasma, equil, fbm, spec=spec, npa=npa, cfpd = cfpd_geo)

    return 0

#%% main script to run as default
def main():
    parser = argparse.ArgumentParser(description="Creates a FIDASIM test case")

    parser.add_argument('path', help='Result directory')
    parser.add_argument('-r', '--runid', default = 'test', help='Test runid')

    args = parser.parse_args()

    run_main_setup(args)

if __name__=='__main__':
    main()
