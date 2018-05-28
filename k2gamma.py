#!/usr/bin/env python

'''
Convert the k-sampled MO to corresponding Gamma-point supercell MO.
Zhihao Cui zcui@caltech.edu
'''

import numpy as np
import scipy
import cmath, os
from pyscf import lib
import sys 

from pyscf.lib import numpy_helper as np_helper
import copy


#def list_pop(lst, idx):
#    lst_copy = copy.deepcopy(lst)
#    p = lst_copy.pop(idx)
#    return p, lst_copy


def get_R_vec(mol, abs_kpts, kmesh):

    '''
    get supercell R vector based on k mesh
    '''

    latt_vec = cell.lattice_vectors()
    R_rel_a = np.arange(kmesh[0]) 
    R_rel_b = np.arange(kmesh[1]) 
    R_rel_c = np.arange(kmesh[2])
    R_rel_mesh = np_helper.cartesian_prod((R_rel_a, R_rel_b, R_rel_c))
    R_abs_mesh = np.einsum('nu, uv -> nv', R_rel_mesh, latt_vec)
    return R_abs_mesh


def null(A, eps=5e-4):

    '''
    return the null space of matrix A, tolerance is eps
    '''
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T


def find_degenerate(mo_energy, mo_coeff, real_skip = False, tol = 1e-5):

    '''
    split the mo_energy into groups based on degenracy.
    further split the real MO out by set real_skip to be True
    mo_energy should range from lowest to highest.
    return grouped mo_energy and its indices.    
    '''

    
    real_tol = tol * 0.01
    
    res = []
    res_idx = []
    g_cur = [mo_energy[0]]
    idx_cur = [0]

    for i in xrange(1, len(mo_energy)):
        diff = mo_energy[i] - mo_energy[i-1]
        if diff < tol:
            g_cur.append(mo_energy[i])
            idx_cur.append(i)
        else:
            res.append(g_cur)
            res_idx.append(idx_cur)
            g_cur = [mo_energy[i]]
            idx_cur = [i]
        if i == len(mo_energy)-1:
            res.append(g_cur)
            res_idx.append(idx_cur)

    if real_skip:
        res_idx_new = []
        res_new = []
        for i in xrange(len(res_idx)):
            res_idx_tmp = copy.deepcopy(res_idx[i])
            res_tmp = copy.deepcopy(res[i])
            
            tmp = 0            

            for j in xrange(len(res_idx_tmp)):
                if np.linalg.norm(mo_coeff[:,res_idx_tmp[j-tmp]].imag) < real_tol:
                    p = res_idx_tmp.pop(j-tmp)
                    res_idx_new.append([p])

                    e = res_tmp.pop(j-tmp)
                    res_new.append([e])

                    tmp += 1

            if res_idx_tmp != []:
                    res_idx_new.append(res_idx_tmp) 
                    res_new.append(res_tmp)
                
        # sort again to make sure the slightly lower energy state to be the first
        sort_idx = sorted(range(len(res_new)), key=lambda k: res_new[k])
        res_new = [res_new[i] for i in sort_idx]
        res_idx_new = [res_idx_new[i] for i in sort_idx]
        res_idx = res_idx_new
        res = res_new

    return res, res_idx


def k2gamma(kmf, abs_kpts, kmesh, realize = True, tol_deg = 1e-5):

    '''
    convert the k-sampled mo coefficient to corresponding supercell gamma-point mo coefficient.
    set realize = True to make sure the final wavefunction to be real.
    math:
         C_{\nu ' n'} = C_{\vecR\mu, \veck m} = \qty[ \frac{1}{\sqrt{N_{\UC}}} \e^{\ii \veck\cdot\vecR} C^{\veck}_{\mu  m}]
    '''

    np.set_printoptions(4,linewidth=1000)

    R_abs_mesh = get_R_vec(kmf.cell, abs_kpts, kmesh)    
    phase = np.exp(-1j*np.einsum('ru, ku -> rk',R_abs_mesh, abs_kpts))
    
    E_k = np.asarray(kmf.mo_energy)
    C_k = np.asarray(kmf.mo_coeff)
    Nk, Nao, Nmo = C_k.shape
    NR = R_abs_mesh.shape[0]    

    C_gamma = np.einsum('kR, kMm -> RMkm', phase, C_k) / np.sqrt(NR)
    E_k_sort_idx = np.unravel_index(np.argsort(E_k, axis=None), E_k.shape)
    E_k_sort = E_k[E_k_sort_idx]   
    
    C_gamma_tmp = np.zeros((NR, Nao, Nk*Nmo), dtype=np.complex)
    
    for R in xrange(NR):
        for mu in xrange(Nmo):
            C_gamma_tmp[R, mu, :] = C_gamma[R, mu][E_k_sort_idx]
 
    C_gamma = C_gamma_tmp.reshape((NR*Nao, Nk*Nmo))
    
    if realize:
        print "realize the gamma point MO ..." 
        C_gamma_real = np.zeros_like(C_gamma, dtype = np.double)

        res, res_idx = find_degenerate(E_k_sort, C_gamma, real_skip = True, tol = tol_deg )
        print "energy spectrum group:", res
        print "energy idx:", res_idx
        col_idx = 0
        for i, gi_idx in enumerate(res_idx):
            gi = C_gamma[:,gi_idx]
            #if len(gi_idx) == 1:
            #    C_gamma_real[:, [col_idx]] = gi.real
            #    col_idx += 1
            #    continue
            null_coeff_real = null(gi.real)
            null_coeff_imag = null(gi.imag)
            if null_coeff_real.shape[1] + null_coeff_imag.shape[1] != len(gi_idx):
                print "realization error"
                sys.exit(1)
            for j in xrange(null_coeff_real.shape[1]): 
                C_gamma_real[:,col_idx] = gi.dot(null_coeff_real[:,j]).imag
                col_idx += 1
            for j in xrange(null_coeff_imag.shape[1]): 
                C_gamma_real[:,col_idx] = gi.dot(null_coeff_imag[:,j]).real
                col_idx += 1
        C_gamma = C_gamma_real
    
    # TODO orthonomalize
        
    return C_gamma

            
if __name__ == '__main__':

    import numpy as np
    from pyscf import scf, gto
    from pyscf.pbc import gto as pgto
    from pyscf.pbc import scf as pscf

    cell = pgto.Cell()
    cell.atom = '''
    H 0.  0.  0.
    H 0.8 0.0 0.0
    '''
#    cell.atom = '''
#     C                  3.17500000    3.17500000    3.17500000
#     H                  2.54626556    2.54626556    2.54626556
#     H                  3.80373444    3.80373444    2.54626556
#     H                  2.54626556    3.80373444    3.80373444
#     H                  3.80373444    2.54626556    3.80373444
#    '''

    cell.basis = 'sto3g'
    #cell.a = np.eye(3) * 10.35
    cell.a = np.array([[1.6, 0.0, 0.0], [0.0, 20.0, 0.0],[0.0, 0.0, 20.0]])
    cell.mesh = [201, 201, 201 ]
    cell.verbose = 6
    #cell.chkfile = './chkfile'
    cell.unit='A'
    cell.build()


    kmesh = [3, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    #print abs_kpts
    scaled_kpts = cell.get_scaled_kpts(abs_kpts)
    #print scaled_kpts

    kmf = pscf.KRHF(cell, abs_kpts).density_fit()
    #kmf = pscf.KRHF(cell, abs_kpts)
    #kmf.__dict__.update(scf.chkfile.load('ch4.chk', 'scf')) # test
    ekpt = kmf.run()
    print k2gamma(kmf, abs_kpts, kmesh)
