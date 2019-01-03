#!/usr/bin/env python

import os
import sys
import math

import voxel_array

import pandas as pd
import numpy as np
import warnings

# Useful numbers
# N [-1.45837285,  0 , 0]
# CA [0., 0., 0.]
# C [0.55221403, 1.41890368, 0.        ]
# CB [ 0.52892494, -0.77445692, -1.19923854]


N = 0
CA = 1
CB = 2
C = 3
O = 4
R = 5

ATOM_NAMES=['N', 'CA', 'CB', 'C', 'O']
assert(R == len(ATOM_NAMES))

def readpdb(fname):
    n = 'het ai an rn ch ri x y z occ bfac elem'.split()
    w = (6, 5, 5, 4, 2, 4, 12, 8, 8, 6, 6, 99)
    assert len(n) is len(w)
    compression = "gzip" if fname.endswith(".gz") else None
    df = pd.read_fwf(fname, widths=w, names=n, compression=compression)
    df = df[np.logical_or(df.het == 'ATOM', df.het == 'HETATM')]
    df.het = df.het == 'HETATM'
    df.ai = df.ai.astype('i4')
    # df.an = df.an.astype('S4')  
    # df.rn = df.rn.astype('S3')  
    # df.ch = df.ch.astype('S1')  
    df.ri = df.ri.astype('i4')
    df.x = df.x.astype('f4')
    df.y = df.y.astype('f4')
    df.z = df.z.astype('f4')
    df.occ = df.occ.astype('f4')
    df.bfac = df.bfac.astype('f4')
    # df.elem = df.elem.astype('S4')  
    return df


# N CA C O CB
# assumes unique ascending res numbers
warnings.filterwarnings("ignore", 'This pattern has match groups')
def npose_from_file(fname):
    pdpose = readpdb(fname)

    # This evaluates to a regex that looks like this "^(N|CA|CB|C|O)$"
    just_my_atoms = pdpose[pdpose.an.str.contains("^(%s)$"%('|'.join(ATOM_NAMES)))].copy()

    # Put the atoms in the order we need them
    just_my_atoms['ind'] = just_my_atoms['ri']*R
    for i, atom in enumerate(ATOM_NAMES):
        just_my_atoms['ind'] += (just_my_atoms['an'] == ATOM_NAMES[i])*i

    just_my_atoms = just_my_atoms.sort_values('ind')

    # Fix glycines
    gb = just_my_atoms.groupby("ri")
    is_gly = gb.apply(lambda x: len(x) == R-1)

    cb_rows = []
    for ri in is_gly[is_gly].index:
        res_df = just_my_atoms[just_my_atoms['ri'] == ri]
        n_ca_c_df = res_df[res_df.an.str.contains("^(N|CA|C)$")]
        n_ca_c = n_ca_c_df[['x', 'y', 'z']].values.astype('f8')
        xform = get_stub_from_n_ca_c(n_ca_c[0], n_ca_c[1], n_ca_c[2])
        cb = get_CB_from_xform(xform)
        cb_row = n_ca_c_df.iloc[1].copy()
        cb_row['an'] = 'CB'
        cb_row['x'] = cb[0]
        cb_row['y'] = cb[1]
        cb_row['z'] = cb[2]
        cb_row['ind'] = cb_row['ri']*R+CB
        cb_rows.append(cb_row)

    if ( len(cb_rows) > 0 ):
        just_my_atoms = just_my_atoms.append(cb_rows)
        just_my_atoms = just_my_atoms.sort_values('ind')


    nres = len(just_my_atoms.ri.unique())
    # This will trigger if you have multiple res with the same resi for instance
    assert( len(just_my_atoms) / nres == R )

    npose = just_my_atoms[['x', 'y', 'z', 'z']].values.astype('f8')
    npose[:,3] = 1.0

    return npose

def nsize(npose):
    return int(len(npose)/R)

def tsize(tpose):
    return len(tpose)

def itsize(itpose):
    return len(itpose)

def get_res( npose, resnum):
    return npose[R*resnum:R*(resnum+1)]

def get_stub_from_n_ca_c(n, ca, c):
    e1 = ca - n
    e1 /= np.linalg.norm(e1)

    e3 = np.cross( e1, c - n )
    e3 /= np.linalg.norm(e3)

    e2 = np.cross( e3, e1 )

    stub = np.zeros((4, 4))
    stub[...,:3,0] = e1
    stub[...,:3,1] = e2
    stub[...,:3,2] = e3
    stub[...,:3,3] = ca
    stub[...,3,3] = 1.0

    return stub

def get_stub_from_npose(npose, resnum):
    # core::kinematics::Stub( CA, N, C )

    res = get_res(npose, resnum)

    return get_stub_from_n_ca_c(res[N,:3], res[CA,:3], res[C,:3])



_atom_record_format = (
    "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}\n"
)


def format_atom(
        atomi=0,
        atomn='ATOM',
        idx=' ',
        resn='RES',
        chain='A',
        resi=0,
        insert=' ',
        x=0,
        y=0,
        z=0,
        occ=1,
        b=0
):
    return _atom_record_format.format(**locals())

_atom_names = []
for name in ATOM_NAMES:
    this_name = " " + name + "       "
    _atom_names.append(this_name[:4])

def dump_npdb(npose, fname):
    with open(fname, 'w') as out:
        for ri, res in enumerate(npose.reshape(-1, R, 4)):
            atom_offset = ri*R+1
            for i, atomi in enumerate([N, CA, C, O, CB]):
                a = res[atomi]
                out.write( format_atom(
                    atomi=atom_offset+i,
                    resn='ALA',
                    resi=ri+1,
                    atomn=_atom_names[atomi],
                    x=a[0],
                    y=a[1],
                    z=a[2]
                    ))

def xform_to_superimpose_nposes( mobile, mobile_resnum, ref, ref_resnum ):

    mobile_stub = get_stub_from_npose(mobile, mobile_resnum)
    mobile_stub_inv = np.linalg.inv(mobile_stub)

    ref_stub = get_stub_from_npose(ref, ref_resnum)

    xform = ref_stub @ mobile_stub_inv

    return xform

def xform_npose(xform, npose):
    return (xform @ npose[...,None]).reshape(-1, 4)


def extract_N_CA_C(npose):
    indices = []
    for i in range(nsize(npose)):
        indices.append(i*R+N)
        indices.append(i*R+CA)
        indices.append(i*R+C)
    return npose[indices]

def extract_CA(npose):
    indices = np.arange(CA, nsize(npose)*R, R)
    return npose[indices]

def points_from_tpose(tpose):
    return tpose[:,:,-1]


def calc_rmsd(npose1, npose2):
    assert( len(npose1) == len(npose2))
    return math.sqrt(np.sum(np.square(np.linalg.norm(npose1[:,:-3] - npose2[:,:-3], axis=0))) / ( len(npose1) ))

def tpose_from_npose( npose ):
    tpose = []
    for i in range(nsize(npose)):
        tpose.append(get_stub_from_npose(npose, i))
    return np.stack(tpose)

def itpose_from_tpose( tpose ):
    itpose = []
    for tres in tpose:
        itpose.append(np.linalg.inv(tres))
    return np.stack(itpose)



def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string


def get_tag(fname):
    name = os.path.basename(fname)
    return my_rstrip(my_rstrip(name, ".gz"), ".pdb")


#Bounds are lb, ub, resl
def clashgrid_from_npose(npose):#, bounds=None, add_to_this=None):
    return clashgrid_from_points( extract_CA(npose))#, add_to_this)

#Bounds are lb, ub, resl
def clashgrid_from_tpose(tpose):#, bounds=None, add_to_this=None):
    return clashgrid_from_points( points_from_tpose(tpose))#, add_to_this)

_resl = 0.5
_atom_size = 5 - _resl
def clashgrid_from_points(points):
    points = points[:,:3]
    low = np.min(points, axis=0) - _atom_size - _resl*2
    high = np.max(points, axis=0) + _atom_size + _resl*2

    clashgrid = voxel_array.VoxelArray(low, high, np.array([_resl]*3), bool)

    for pt in points:
        inds = clashgrid.indices_within_x_of(_atom_size, pt)
        clashgrid.arr[tuple(inds.T)] = True

    return clashgrid

def xforms_from_four_points(c, u, v, w):
    c = c[...,:3]
    u = u[...,:3]
    v = v[...,:3]
    w = w[...,:3]

    e1 = u - v
    e1 = e1 / np.linalg.norm(e1, axis=1)[...,None]
    e3 = np.cross( e1, w - v, axis=1)
    e3 = e3 / np.linalg.norm(e3, axis=1)[...,None]
    e2 = np.cross(e3, e1, axis=1)

    xforms = np.zeros((len(c), 4, 4))
    xforms[...,:3,0] = e1
    xforms[...,:3,1] = e2
    xforms[...,:3,2] = e3
    xforms[...,:3,3] = c
    xforms[...,3,3] = 1.0

    return xforms

def npose_to_motif_hash_frames(npose):
    by_res = npose.reshape(-1, R, 4)


    Ns = by_res[:,N]
    CAs = by_res[:,CA]
    Cs = by_res[:,C]

    CEN = np.array([-0.865810,-1.764143,1.524857, 1.0])


    #CEN = Xform().from_four_points( CA, N, CA, C ) * CEN;
    # Vec const DIR1 = C-N;
    # Vec const CEN2 = (C+N)/2;
    # return Xform().from_four_points( CEN, CEN2, CA, CA+DIR1 );

    cen = (xforms_from_four_points(CAs, Ns, CAs, Cs) @ CEN[...,None]).reshape(-1, 4)

    dir1 = Cs - Ns
    cen2 = (Cs+Ns)/2

    return xforms_from_four_points(cen, cen2, CAs, CAs+dir1)


def pair_xform(xform1, xform2):
    return np.linalg.inv(xform1) @ xform2

def sin_cos_range( x, tol=0.001):
    if ( x >= -1 and x <= 1 ):
        return x
    elif ( x <= -1 and x >= -( 1 + tol ) ):
        return -1
    elif ( x >= 1 and x <= 1 + tol ):
        return 1
    else:
        eprint("sin_cos_range ERROR: %.8f"%x )
        return -1 if x < 0 else 1

_xx = (0, 0)
_xy = (0, 1)
_xz = (0, 2)
_yx = (1, 0)
_yy = (1, 1)
_yz = (1, 2)
_zx = (2, 0)
_zy = (2, 1)
_zz = (2, 2)
_x = (0, 3)
_y = (1, 3)
_z = (2, 3)

_float_precision = 0.00001

# def rt6_from_xform(xform, rt6):
#     rt6[1] = xform[0,3]
#     rt6[2] = xform[1,3]
#     rt6[3] = xform[2,3]

#     if ( xform[_zz] >= 1 - _float_precision ):
#         e1 = math.atan2( sin_cos_range( xform[_yx] ), sin_cos_range( xform[_xx] ) )
#         e2 = 0
#         e3 = 0
#     elif ( xform[_zz] <= -1 + _float_precision ):
#         e1 = math.atan2( sin_cos_range( xform[_yx] ), sin_cos_range( xform[_xx] ) )
#         e2 = 0
#         e3 = math.pi
#     else:
#         pos_sin_theta = math.sqrt( 1 - xform[_zz]**2 )
#         e3 = math.asin( pos_sin_theta )
#         if ( xform[_zz] < 0 ):
#             e3 = math.pi - e3
#         e1 = math.atan2( xform[_xz], -xform[_yz])
#         e2 = math.atan2( xform[_zx],  xform[_zy])

#     if ( e1 < 0 ):
#         e1 += math.pi * 2
#     if ( e2 < 0 ):
#         e2 += math.pi * 2

#     rt6[4] = 180/math.pi*min(max(0, e1), math.pi*2-0.0000000000001)
#     rt6[5] = 180/math.pi*min(max(0, e2), math.pi*2-0.0000000000001)
#     rt6[6] = 180/math.pi*min(max(0, e3), math.pi  -0.0000000000001)


#     return rt6

def rt6_from_xform(xform, xyzTransform):

    xyzTransform.R.xx = xform[_xx]
    xyzTransform.R.xy = xform[_xy]
    xyzTransform.R.xz = xform[_xz]
    xyzTransform.R.yx = xform[_yx]
    xyzTransform.R.yy = xform[_yy]
    xyzTransform.R.yz = xform[_yz]
    xyzTransform.R.zx = xform[_zx]
    xyzTransform.R.zy = xform[_zy]
    xyzTransform.R.zz = xform[_zz]
    xyzTransform.t.x = xform[_x]
    xyzTransform.t.y = xform[_y]
    xyzTransform.t.z = xform[_z]

    return xyzTransform.rt6()


def xform_from_axis_angle_deg( axis, angle ):
    return xform_from_axis_angle_rad( axis, angle * math.pi / 180 )

def xform_from_axis_angle_rad( axis, angle ):
    xform = np.zeros((4, 4))
    xform[3,3] = 1.0

    cos = math.cos(angle)
    sin = math.sin(angle)
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    xform[0, 0] = cos + ux**2*(1-cos)
    xform[0, 1] = ux*uy*(1-cos) - uz*sin
    xform[0, 2] = ux*uz*(1-cos) + uy*sin

    xform[1, 0] = uy*ux*(1-cos) + uz*sin
    xform[1, 1] = cos + uy**2*(1-cos)
    xform[1, 2] = uy*uz*(1-cos) - ux*sin

    xform[2, 0] = uz*ux*(1-cos) - uy*sin
    xform[2, 1] = uz*uy*(1-cos) + ux*sin
    xform[2, 2] = cos + uz**2*(1-cos)

    return xform


def get_N_from_xform(xform):
    N_pos = np.array([-1.45837285,  0, 0, 1.0])
    return xform @ N_pos

def get_C_from_xform(xform):
    C_pos = np.array([0.55221403, 1.41890368, 0, 1.0])
    return xform @ C_pos

def get_CB_from_xform(xform):
    CB_pos = np.array([0.52892494, -0.77445692, -1.19923854, 1.0])
    return xform @ CB_pos

def get_phi_vector(xform):
    N_pos = get_N_from_xform(xform)
    return xform[:3,3] - N_pos[:3]

def get_psi_vector(xform):
    C_pos = get_C_from_xform(xform)
    return C_pos[:3] - xform[:3,3]

def get_phi_rotation_xform(xform, angle_deg, ca):
    vec = get_phi_vector(xform)
    vec /= np.linalg.norm(vec)
    rotate_xform = xform_from_axis_angle_deg(vec, angle_deg)
    trans = ca[:3] + (rotate_xform @ -ca)[:3]
    rotate_xform[:3,3] = trans
    return rotate_xform

def get_psi_rotation_xform(xform, angle_deg, ca):
    vec = get_psi_vector(xform)
    vec /= np.linalg.norm(vec)
    rotate_xform = xform_from_axis_angle_deg(vec, angle_deg)
    trans = ca[:3] + (rotate_xform @ -ca)[:3]
    rotate_xform[:3,3] = trans
    return rotate_xform

def apply_dihedral_to_points(points, xform, start_pos):
    unaffected = points[:start_pos]
    modified = xform_npose(xform, points[start_pos:])
    return np.concatenate([unaffected, modified])

def apply_dihedral_to_xforms(xforms, xform, start_pos):
    unaffected = xforms[:start_pos]
    modified = xform @ xforms[start_pos:]
    return np.concatenate([unaffected, modified])

def rotate_npose_phi(npose, tpose, resno, delta_phi):
    phi_xform = get_phi_rotation_xform(tpose[resno], delta_phi, npose[resno*R+CA])
    return apply_dihedral_to_points(npose, phi_xform, resno*R+CA)

def rotate_tpose_phi(tpose, resno, delta_phi):
    phi_xform = get_phi_rotation_xform(tpose[resno], delta_phi, tpose[resno,:,3])
    return apply_dihedral_to_xforms(tpose, phi_xform, resno)    # This affects resno xform

def rotate_npose_psi(npose, tpose, resno, delta_psi):
    psi_xform = get_psi_rotation_xform(tpose[resno], delta_psi, npose[resno*R+CA])
    return apply_dihedral_to_points(npose, psi_xform, resno*R+C)

def rotate_tpose_psi(tpose, resno, delta_psi):
    psi_xform = get_psi_rotation_xform(tpose[resno], delta_psi, tpose[resno,:,3])
    return apply_dihedral_to_xforms(tpose, psi_xform, resno+1)

def set_npose_phi(npose, tpose, phis, resno, phi):
    return set_phi(npose, tpose, phis, resno, phi, R, CA, CA, 0)

def set_ca_phi(points, tpose, phis, resno, phi):
    return set_phi(points, tpose, phis, resno, phi, 1, 0, 0, 0)    # This affects resno

def set_phi(points, tpose, phis, resno, phi, local_R, R_off, R_CA, t_off):
    delta_phi = phi - phis[resno]
    new_phis = phis.copy()
    new_phis[resno] = phi

    phi_xform = get_phi_rotation_xform(tpose[resno], delta_phi, points[resno*local_R+R_CA])
    new_points = apply_dihedral_to_points(points, phi_xform, resno*local_R+R_off)
    new_tpose = apply_dihedral_to_xforms(tpose, phi_xform, resno+t_off)

    return new_points, new_tpose, new_phis

def set_npose_psi(npose, tpose, psis, resno, psi):
    return set_psi(npose, tpose, psis, resno, psi, R, C, CA, 1)

def set_ca_psi(points, tpose, psis, resno, phi):
    return set_psi(points, tpose, psis, resno, psi, 1, 1, 0, 1)

def set_psi(points, tpose, psis, resno, psi, local_R, R_off, R_CA, t_off):
    delta_psi = psi - psis[resno]
    new_psis = psis.copy()
    new_psis[resno] = psi

    psi_xform = get_psi_rotation_xform(tpose[resno], delta_psi, points[resno*local_R+R_CA])
    new_points = apply_dihedral_to_points(points, psi_xform, resno*local_R+R_off)
    new_tpose = apply_dihedral_to_xforms(tpose, psi_xform, resno+t_off)

    return new_points, new_tpose, new_psis


def get_dihedral(atom1, atom2, atom3, atom4):
    a = atom2 - atom1
    a /= np.linalg.norm(a)
    b = atom3 - atom2
    b /= np.linalg.norm(b)
    c = atom4 - atom3
    c /= np.linalg.norm(c)

    x = -np.dot( a, c ) + ( np.dot( a, b ) * np.dot( b, c) )
    y = np.dot( a, np.cross( b, c ) )

    angle = 0 if ( y == 0 and x == 0 ) else math.atan2( y, x )

    return angle


def get_npose_phis(npose):
    phis = []
    phis.append(0)

    for i in range(1, nsize(npose)):
        offset = i * R
        phis.append(180/math.pi * get_dihedral( npose[offset-R+C,:3], 
                                                npose[offset+N,:3],
                                                npose[offset+CA,:3],
                                                npose[offset+C,:3] 
                                               ))

    return np.array(phis)

def get_npose_psis(npose):
    psis = []

    for i in range(nsize(npose) - 1):
        offset = i * R
        psis.append(180/math.pi * get_dihedral( npose[offset-N,:3], 
                                                npose[offset+CA,:3],
                                                npose[offset+C,:3],
                                                npose[offset+R+N,:3] 
                                               ))

    psis.append(0)
    return np.array(psis)











