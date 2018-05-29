#!/usr/bin/env python

'''
Convert the k-sampled MO to corresponding Gamma-point supercell MO.
Zhihao Cui zcui@caltech.edu
'''

import numpy as np
import scipy
from scipy import linalg as la
import cmath, os, sys, copy

from pyscf import lib
from pyscf.lib import numpy_helper as np_helper
from pyscf.pbc.tools import pbc as tools_pbc
from pyscf import scf, gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf


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

def nullspace(A, atol=5e-4, rtol=0):

    '''
    Compute an approximate basis for the nullspace of A.
    http://scipy-cookbook.readthedocs.io/items/RankNullspace.html

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    '''

    A = np.atleast_2d(A)
    u, s, vh = la.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def find_degenerate(mo_energy, mo_coeff, real_split = False, tol = 1e-5):

    '''
    split the mo_energy into groups based on degenracy.
    further split the real MO out by set real_split to be True
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

    if real_split:
        res_idx_new = []
        res_new = []
        for i in xrange(len(res_idx)):
            res_idx_tmp = copy.deepcopy(res_idx[i])
            res_tmp = copy.deepcopy(res[i])
            
            tmp = 0            

            for j in xrange(len(res_idx_tmp)):
                if la.norm(mo_coeff[:,res_idx_tmp[j-tmp]].imag) < real_tol:
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
    phase = np.exp(1j*np.einsum('Ru, ku -> Rk',R_abs_mesh, abs_kpts))
    
    E_k = np.asarray(kmf.mo_energy)
    C_k = np.asarray(kmf.mo_coeff)
    S_k = np.asarray(kmf.get_ovlp())
    S_k_sqrt = [la.sqrtm(S_k[i]) for i in xrange(len(S_k))]
    # lowdin orthogonalization
    C_k = np.asarray([S_k_sqrt[i].dot(C_k[i]) for i in xrange(len(C_k))])
    
    Nk, Nao, Nmo = C_k.shape
    NR = R_abs_mesh.shape[0]    

    C_gamma = np.einsum('Rk, kMm -> RMkm', phase, C_k) / np.sqrt(NR)
    C_gamma = C_gamma.reshape((NR, Nao, Nk*Nmo))

    # sort energy of km
    E_k_flat = E_k.flatten()
    E_k_sort_idx = np.argsort(E_k_flat)
    E_k_sort = E_k_flat[E_k_sort_idx]
    
    C_gamma = C_gamma[:, :, E_k_sort_idx]
    C_gamma = C_gamma.reshape((NR*Nao, Nk*Nmo))
   
    # make MO to be real
    if realize:
        
        real_tol = tol_deg
        print "Realize the gamma point MO ..." 
        C_gamma_real = np.zeros_like(C_gamma, dtype = np.double)

        res, res_idx = find_degenerate(E_k_sort, C_gamma, real_split = True, tol = tol_deg )
        print "Energy spectrum group:", res
        print "Energy idx:", res_idx
        col_idx = 0
        for i, gi_idx in enumerate(res_idx):
            gi = C_gamma[:,gi_idx]
            null_coeff_real = nullspace(gi.real)
            null_coeff_imag = nullspace(gi.imag)

            if null_coeff_real.shape[1] + null_coeff_imag.shape[1] != len(gi_idx):
                print "realization error, not find enough linear combination coefficient"
                sys.exit(1)

            for j in xrange(null_coeff_real.shape[1]):
                gi_after = gi.dot(null_coeff_real[:,j])
                if la.norm(gi_after.real) > real_tol:
                    print "realization error, real part is too large"
                    sys.exit(1)
                C_gamma_real[:,col_idx] = gi_after.imag
                col_idx += 1

            for j in xrange(null_coeff_imag.shape[1]): 
                gi_after = gi.dot(null_coeff_imag[:,j]) 
                if la.norm(gi_after.imag) > real_tol:
                    print "realization error, imag part is too large"
                    sys.exit(1)
                C_gamma_real[:,col_idx] = gi_after.real
                col_idx += 1

        C_gamma = C_gamma_real
    
    # from othonomal MO to non-orthogonal original AO basis
    sc = tools_pbc.super_cell(cell, kmesh)
    sc.verbose = 0
    kmf_sc = pscf.KRHF(sc, [[0.0,0.0,0.0]]).density_fit() # TODO spin polarize ? DF or not?
    S_sc = kmf_sc.get_ovlp()[0]
    # gamma MO not othogonal
    C_gamma = la.inv(la.sqrtm(S_sc)).dot(C_gamma)

    return C_gamma

            
if __name__ == '__main__':

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
    cell.a = np.array([[2.0, 0.0, 0.0], [0.0, 20.0, 0.0],[0.0, 0.0, 20.0]])
    cell.mesh = [51, 51, 51 ]
    cell.verbose = 6
    #cell.chkfile = './chkfile'
    cell.unit='B'
    cell.build()
    
    kmesh = [3, 1, 1]
    abs_kpts = cell.make_kpts(kmesh)
    scaled_kpts = cell.get_scaled_kpts(abs_kpts)

    kmf = pscf.KRHF(cell, abs_kpts).density_fit()
    #kmf = pscf.KRHF(cell, abs_kpts)
    #kmf.__dict__.update(scf.chkfile.load('ch4.chk', 'scf')) # test
    ekpt = kmf.run()

    c_g_ao = k2gamma(kmf, abs_kpts, kmesh)
    print "gamma MO in supercell AO basis:"
    print c_g_ao
  
    # check whether the MO is correctly coverted: 
    sc = tools_pbc.super_cell(cell, kmesh)
    sc.verbose = 0
    kmf_sc = pscf.KRHF(sc, [[0.0,0.0,0.0]]).density_fit()
    #s = scf.hf.get_ovlp(sc) # NOTE what is the diff?
    s = kmf_sc.get_ovlp()[0]
    print "Run supercell gamma point calculation..." 
    ekpt_sc = kmf_sc.run()
    sc_mo = kmf_sc.mo_coeff[0]
    print "supercell MO from calculation:"
    print sc_mo

    # lowdin of sc_mo and c_g_ao
    sc_mo_o = la.sqrtm(s).dot(sc_mo)
    c_g = la.sqrtm(s).dot(c_g_ao)
 
    # do some linear combination of degenerate states to make the converted MO looks like the direct calculated one
    u,sigma,v = la.svd(c_g[:, [1,2]].T.dot(sc_mo_o[:,[1,2]]))
    lc1 = c_g[:, [1,2]].dot(u[:, 0])
    lc2 = c_g[:, [1,2]].dot(u[:, 1])

    c_g[:,1] = lc1
    c_g[:,2] = lc2   
 
    u,sigma,v = la.svd(c_g[:, [3,4]].T.dot(sc_mo_o[:,[3,4]]))
    lc1 = c_g[:, [3,4]].dot(u[:, 0])
    lc2 = c_g[:, [3,4]].dot(u[:, 1])

    c_g[:,3] = lc1
    c_g[:,4] = lc2     

    print "converted gamma MO in AO basis, after some linear combination of degenerate state:"
    c_g_ao = la.inv(la.sqrtm(s)).dot(c_g)
    print c_g_ao


