#!/usr/bin/env python

import os
import sys
import math

import voxel_array

import pandas as pd
import numpy as np
import warnings

try:
    from numba import njit
    from numba import jit
    import numba
except:
    sys.path.append("/home/bcov/sc/random/just_numba")
    from numba import njit

# Useful numbers
# N [-1.45837285,  0 , 0]
# CA [0., 0., 0.]
# C [0.55221403, 1.41890368, 0.        ]
# CB [ 0.52892494, -0.77445692, -1.19923854]

if ( hasattr(os, 'ATOM_NAMES') ):
    assert( hasattr(os, 'PDB_ORDER') )

    ATOM_NAMES = os.ATOM_NAMES
    PDB_ORDER = os.PDB_ORDER
else:
    ATOM_NAMES=['N', 'CA', 'CB', 'C', 'O']
    PDB_ORDER = ['N', 'CA', 'C', 'O', 'CB']

_byte_atom_names = []
_atom_names = []
for i, atom_name in enumerate(ATOM_NAMES):
    long_name = " " + atom_name + "       "
    _atom_names.append(long_name[:4])
    _byte_atom_names.append(atom_name.encode())

    globals()[atom_name] = i

R = len(ATOM_NAMES)

if ( "N" not in globals() ):
    N = -1
if ( "C" not in globals() ):
    C = -1
if ( "CB" not in globals() ):
    CB = -1


_pdb_order = []
for name in PDB_ORDER:
    _pdb_order.append( ATOM_NAMES.index(name) )


_space = " ".encode()[0]
@njit(fastmath=True)
def space_strip(string):
    start = 0
    while(start < len(string) and string[start] == _space):
        start += 1
    end = len(string)
    while( end > 0 and string[end-1] == _space):
        end -= 1
    return string[start:end]

@njit(fastmath=True)
def byte_startswith( haystack, needle ):
    if ( len(haystack) < len(needle) ):
        return False
    for i in range(len(needle)):
        if ( haystack[i] != needle[i] ):
            return False
    return True

@njit(fastmath=True)
def byte_equals( haystack, needle ):
    if ( len(haystack) != len(needle) ):
        return False
    for i in range(len(needle)):
        if ( haystack[i] != needle[i] ):
            return False
    return True

@njit(fastmath=True)
def getline( bytess, start ):
    cur = start
    if ( start + 78 < len(bytess) ):
        if ( bytess[start+78] == 10 ):
            return bytess[start:start+79], start+79
    while (cur < len(bytess) and bytess[cur] != 10 ):
        cur += 1
    cur += 1
    return bytess[start:cur], cur

# ord(" ") == 32
# ord("-") == 45
# ord(".") == 46
# ord("0") == 48
# ord("9") == 57

@njit(fastmath=True)
def stof(string):
    multiplier = 0

    parsed = False
    negative = 1
    start = 0
    end = len(string) - 1
    for i in range(len(string)):
        char = string[i]
        # print(char)
        if ( not parsed ):
            if ( char == 32 ): # " "
                start = i + 1
                continue
            if ( char == 45 ): # "-"
                start = i + 1
                negative = -1
                continue
        if ( char == 32 ): # " "
            break
        if ( char == 46 ): # "."
            multiplier = np.float64(1)
            parsed = True
            continue
        if ( char >= 48 and char <= 57 ): # 0 9
            parsed = True
            multiplier /= np.float64(10)
            end = i
            continue
        print("Float parse error! Unrecognized character: ", char)
        assert(False)

    if ( not parsed ):
        print("Float parse error!")
        assert(False)

    result = np.float64(0)

    if ( multiplier == 0 ):
        multiplier = 1
    for i in range(end, start-1, -1):
        char = string[i]
        if ( char == 46 ): # "."
            continue
        value = np.float64(char - 48) # 0

        result += value * multiplier
        multiplier *= np.float64(10)

    result *= negative

    return np.float32(result)



# _atom = "ATOM".encode()
_null_line = "ATOMCB00000000000000000000000000000000000000000000000000000000"#.encode()
_null_line_size = len(_null_line)

# _CB = "CB".encode()
# _empty_bytes = "".encode()

# Switches to next residue whenever
#  Line doesn't start with atom
#  Line isn't long enough
#  Res/resnum/chain changes
@njit(fastmath=True)
def read_npose_from_data( data, null_line_atom_names, scratch_residues):


    _null_line = null_line_atom_names[:_null_line_size]
    array_byte_atom_names = null_line_atom_names[_null_line_size:]

    _CB = _null_line[4:6]
    _atom = _null_line[:4]
    _empty_bytes = _null_line[:0]

    byte_atom_names = []
    for i in range(len(array_byte_atom_names)//4):
        aname = array_byte_atom_names[i*4:i*4+4]
        byte_atom_names.append(space_strip(aname))

    seqpos = 0
    res_ident = _empty_bytes
    res_has_n_atoms = 0
    next_res = False
    scratch_residues[0].fill(0)

    # lines.append(_nulline)

    cursor = 0
    keep_going = True
    while keep_going:
        line, cursor = getline(data, cursor)
        if ( cursor >= len(data) ):
            keep_going = False
            line = _null_line

        # print(iline)

        if ( not byte_startswith(line, _atom) or len(line) < 54 ):
            next_res = True
            res_ident = _empty_bytes
            continue

        ident = line[17:26]
        if ( not byte_equals( ident, res_ident ) ):
            next_res = True

        if ( next_res ):
            if ( res_has_n_atoms > 0 ):

                res = scratch_residues[seqpos]
                if ( res_has_n_atoms != R ):
                    missing = np.where(res[:,3] == 0)[0]

                    # We only know how to fix missing CB
                    first_missing = byte_atom_names[missing[0]]
                    if ( len(missing) > 1 or not byte_equals(first_missing,  _CB) ):
                        err_string = "missing atoms: "
                        # for i in range(len(missing)):
                        #     if ( i != 0 ):
                        #         err_string = err_string + ", "
                        #     err_string = err_string + byte_atom_names[missing[i]]

                        # err_string = err_string + " in residue: " + res_ident + " before line: "
                        # err_string = err_string + str(iline)
                        assert(False)

                    # Fixing CB
                    xform = get_stub_from_n_ca_c(res[N,:3], res[CA,:3], res[C,:3])
                    res[CB] = get_CB_from_xform( xform )


                seqpos += 1
                #If we run out of scratch, double its size
                if ( seqpos == len(scratch_residues) ):
                    old_size = len(scratch_residues)
                    new_scratch = np.zeros((old_size*2, R, 4), np.float32)
                    for i in range(old_size):
                        new_scratch[i] = scratch_residues[i]
                    scratch_residues = new_scratch

                scratch_residues[seqpos].fill(0)

            res_ident = ident
            res_has_n_atoms = 0
            next_res = False

        # avoid parsing stuff we know we don't need
        if ( res_has_n_atoms == R ):
            continue

        atom_name = space_strip(line[12:16])

        # figure out which atom we have
        atomi = -1
        for i in range( R ):
            if ( byte_equals( atom_name, byte_atom_names[i] ) ):
                atomi = i
                break
        if ( atomi == -1 ):
            continue

        res = scratch_residues[seqpos]
        if ( res[atomi,3] != 0 ):
            err_string = "duplicate atom: " #+ atom_name + " in residue: " + res_ident + " at line: " #+ str(iline)
            assert(False)

        res_has_n_atoms += 1

        res[atomi,0] = stof(line[30:38])
        res[atomi,1] = stof(line[38:46])
        res[atomi,2] = stof(line[46:54])
        res[atomi,3] = 1

    to_ret = np.zeros((seqpos, R, 4))
    for i in range(seqpos):
        to_ret[i] = scratch_residues[i]
    
    return to_ret.reshape(-1, 4), scratch_residues


g_scratch_residues = np.zeros((1000,R,4), np.float32)
# for i in range(1000):
#     g_scratch_residues.append(np.zeros((R,4), np.float32))

_array_byte_atom_names = list(" "*len(_byte_atom_names)*4)
for i in range(len(_byte_atom_names)):
    name = _atom_names[i]
    for j in range(len(name)):
        _array_byte_atom_names[i*4+j] = name[j]
_array_atom_names = "".join(_array_byte_atom_names)

_null_line_atom_names = (_null_line + _array_atom_names).encode()

def npose_from_file_fast(fname):
    with open(fname, "rb") as f:
        data = f.read()

    global g_scratch_residues

    npose, scratch = read_npose_from_data( data, _null_line_atom_names,  g_scratch_residues)

    g_scratch_residues = scratch
    return npose.astype(np.float32) # get rid of random numba noise

def readpdb(fname):
    n = 'het ai an rn ch ri x y z occ bfac elem'.split()
    w = (6, 5, 5, 4, 2, 4, 12, 8, 8, 6, 6, 99)
    assert len(n) is len(w)
    compression = "gzip" if fname.endswith(".gz") else None
    df = pd.read_fwf(fname, widths=w, names=n, compression=compression)
    df = df.dropna(subset=['x'])

    # df = df[np.logical_or(df.het == 'ATOM', df.het == 'HETATM')]
    # df.het = df.het == 'HETATM' # slow af

    # df.ai = df.ai.astype('i4')
    # # df.an = df.an.astype('S4')  
    # # df.rn = df.rn.astype('S3')  
    # # df.ch = df.ch.astype('S1')  
    # df.ri = df.ri.astype('i4')
    # df.x = df.x.astype('f4')
    # df.y = df.y.astype('f4')
    # df.z = df.z.astype('f4')
    # df.occ = df.occ.astype('f4')
    # df.bfac = df.bfac.astype('f4')
    # # df.elem = df.elem.astype('S4')  
    return df

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

@njit(fastmath=True)
def cross(vec1, vec2):
    result = np.zeros(3)
    a1, a2, a3 = vec1[0], vec1[1], vec1[2]
    b1, b2, b3 = vec2[0], vec2[1], vec2[2]
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


def get_just_my_atoms(pdpose):
    return pdpose[pdpose.an.str.contains("^(%s)$"%('|'.join(ATOM_NAMES)))].copy()

def add_my_resnum(just_my_atoms):
    just_my_atoms['my_resnum'] = (just_my_atoms['ri'] != just_my_atoms['ri'].shift(1).fillna(-1)).cumsum()


def atom_thing(just_my_atoms):

    the_map = {}
    for i, atom in enumerate(ATOM_NAMES):
        the_map[atom] = i

    just_my_atoms['ind'] = just_my_atoms['my_resnum']*R + just_my_atoms['an'].map(the_map)

    return just_my_atoms



def the_sort(just_my_atoms):
    return just_my_atoms.sort_values('ind')

def gb(just_my_atoms):
    gb = just_my_atoms.groupby("my_resnum")
    is_gly = gb.apply(lambda x: len(x) == R-1)
    return is_gly

def for_loop(just_my_atoms, is_gly):

    cb_rows = []
    for resnum in is_gly[is_gly].index:
        res_df = just_my_atoms[just_my_atoms['my_resnum'] == resnum]
        n_ca_c_df = res_df[res_df.an.str.contains("^(N|CA|C)$")]
        n_ca_c = n_ca_c_df[['x', 'y', 'z']].values.astype('f8')
        xform = get_stub_from_n_ca_c(n_ca_c[0], n_ca_c[1], n_ca_c[2])
        cb = get_CB_from_xform(xform)
        cb_row = n_ca_c_df.iloc[1].copy()
        cb_row['an'] = 'CB'
        cb_row['x'] = cb[0]
        cb_row['y'] = cb[1]
        cb_row['z'] = cb[2]
        cb_row['ind'] = cb_row['my_resnum']*R+CB
        cb_rows.append(cb_row)

    return cb_rows

def actual_append(just_my_atoms, cb_rows):
    return just_my_atoms.append(cb_rows, sort)

def actual_sort(just_my_atoms):
    return just_my_atoms.sort_values('ind')

def appenddd(just_my_atoms, cb_rows):
    if ( len(cb_rows) > 0 ):
        just_my_atoms = actual_append(just_my_atoms, cb_rows) #just_my_atoms.append(cb_rows)
        just_my_atoms = actual_sort(just_my_atoms) #just_my_atoms.sort_values('ind')
    return just_my_atoms

def fix_gly(just_my_atoms):
    # gb = just_my_atoms.groupby("my_resnum")
    # is_gly = gb.apply(lambda x: len(x) == R-1)

    is_gly = gb(just_my_atoms)

    cb_rows = for_loop(just_my_atoms, is_gly)
    # cb_rows = []
    # for resnum in is_gly[is_gly].index:
    #     res_df = just_my_atoms[just_my_atoms['my_resnum'] == resnum]
    #     n_ca_c_df = res_df[res_df.an.str.contains("^(N|CA|C)$")]
    #     n_ca_c = n_ca_c_df[['x', 'y', 'z']].values.astype('f8')
    #     xform = get_stub_from_n_ca_c(n_ca_c[0], n_ca_c[1], n_ca_c[2])
    #     cb = get_CB_from_xform(xform)
    #     cb_row = n_ca_c_df.iloc[1].copy()
    #     cb_row['an'] = 'CB'
    #     cb_row['x'] = cb[0]
    #     cb_row['y'] = cb[1]
    #     cb_row['z'] = cb[2]
    #     cb_row['ind'] = cb_row['my_resnum']*R+CB
    #     cb_rows.append(cb_row)

    just_my_atoms = appenddd(just_my_atoms, cb_rows)
    # if ( len(cb_rows) > 0 ):
    #     just_my_atoms = just_my_atoms.append(cb_rows)
    #     just_my_atoms = just_my_atoms.sort_values('ind')
    return just_my_atoms

def crafting(just_my_atoms):
    npose = just_my_atoms[['x', 'y', 'z', 'z']].values.astype('f8')
    npose[:,3] = 1.0
    return npose

# N CA C O CB
# assumes unique ascending res numbers
warnings.filterwarnings("ignore", 'This pattern has match groups')
def npose_from_file(fname):
    pdpose = readpdb(fname)

    # This evaluates to a regex that looks like this "^(N|CA|CB|C|O)$"
    just_my_atoms = pdpose[pdpose.an.str.contains("^(%s)$"%('|'.join(ATOM_NAMES)))].copy()

    # This identifies places where the previous resnum is not equal to this resnum
    # i.e. the first atom of every residue
    just_my_atoms['my_resnum'] = (just_my_atoms['ri'] != just_my_atoms['ri'].shift(1).fillna(-1)).cumsum()

    nres = len(just_my_atoms.my_resnum.unique())


    # Put the atoms in the order we need them
    the_map = {}
    for i, atom in enumerate(ATOM_NAMES):
        the_map[atom] = i

    just_my_atoms['ind'] = just_my_atoms['my_resnum']*R + just_my_atoms['an'].map(the_map)
    just_my_atoms = just_my_atoms.sort_values('ind')


    if ( len(just_my_atoms) / nres != R ):
        # just_my_atoms = fix_gly(just_my_atoms)
        # Fix glycines
        gb = just_my_atoms.groupby("my_resnum")
        is_gly = gb.apply(lambda x: len(x) == R-1)

        cb_rows = []
        for resnum in is_gly[is_gly].index:
            res_df = just_my_atoms[just_my_atoms['my_resnum'] == resnum]
            n_ca_c_df = res_df[res_df.an.str.contains("^(N|CA|C)$")]
            n_ca_c = n_ca_c_df[['x', 'y', 'z']].values.astype('f8')
            xform = get_stub_from_n_ca_c(n_ca_c[0], n_ca_c[1], n_ca_c[2])
            cb = get_CB_from_xform(xform)
            cb_row = n_ca_c_df.iloc[1].copy()
            cb_row['an'] = 'CB'
            cb_row['x'] = cb[0]
            cb_row['y'] = cb[1]
            cb_row['z'] = cb[2]
            cb_row['ind'] = cb_row['my_resnum']*R+CB
            cb_rows.append(cb_row)

        if ( len(cb_rows) > 0 ):
            just_my_atoms = just_my_atoms.append(cb_rows, sort=False)
            just_my_atoms = just_my_atoms.sort_values('ind')


    # nres = len(just_my_atoms.my_resnum.unique())
    # This will tosgger if you have multiple res with the same resi for instance
    assert( len(just_my_atoms) / nres == R )

    # npose = crafting(just_my_atoms)
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

@njit(fastmath=True)
def get_stub_from_n_ca_c(n, ca, c):
    e1 = ca - n
    e1 /= np.linalg.norm(e1)

    e3 = cross( e1, c - n )
    e3 /= np.linalg.norm(e3)

    e2 = cross( e3, e1 )

    stub = np.zeros((4, 4), np.float32)
    stub[...,:3,0] = e1
    stub[...,:3,1] = e2
    stub[...,:3,2] = e3
    stub[...,:3,3] = ca
    stub[...,3,3] = 1.0

    return stub

def get_stubs_from_n_ca_c(n, ca, c):
    e1 = ca - n
    e1 = np.divide( e1, np.linalg.norm(e1, axis=1)[..., None] )

    e3 = np.cross( e1, c - n, axis=1 )
    e3 = np.divide( e3, np.linalg.norm(e3, axis=1)[..., None] )

    e2 = np.cross( e3, e1, axis=1 )

    stub = np.zeros((len(n), 4, 4))
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

def get_stubs_from_npose( npose ):
    ns  = extract_atoms(npose, [N])
    cas = extract_atoms(npose, [CA])
    cs  = extract_atoms(npose, [C])

    return get_stubs_from_n_ca_c(ns[:,:3], cas[:,:3], cs[:,:3])

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


def dump_npdb(npose, fname, atoms_present=list(range(R)), pdb_order=_pdb_order):
    assert(len(atoms_present) == len(pdb_order))
    local_R = len(atoms_present)
    with open(fname, 'w') as out:
        for ri, res in enumerate(npose.reshape(-1, local_R, 4)):
            atom_offset = ri*local_R+1
            for i, atomi in enumerate(pdb_order):
                a = res[atoms_present.index(atomi)]
                out.write( format_atom(
                    atomi=(atom_offset+i)%100000,
                    resn='ALA',
                    resi=(ri+1)%10000,
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

def extract_atoms(npose, atoms):
    return npose.reshape(-1, R, 4)[...,atoms,:].reshape(-1,4)

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
    return get_stubs_from_npose( npose )


def itpose_from_tpose( tpose ):
    return np.linalg.inv(tpose)


def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string


def get_tag(fname):
    name = os.path.basename(fname)
    return my_rstrip(my_rstrip(name, ".gz"), ".pdb")


# _resl = 0.5
# _atom_size = 5 - _resl
#Bounds are lb, ub, resl

#num_clashes = clash_grid.arr[tuple(clash_grid.floats_to_indices(xformed_cas).T)].sum()
def ca_clashgrid_from_npose(npose, atom_size, resl):
    return clashgrid_from_points( extract_CA(npose))

#Bounds are lb, ub, resl
def clashgrid_from_tpose(tpose, atom_size, resl):
    return clashgrid_from_points( points_from_tpose(tpose))

def clashgrid_from_points(points, atom_size, resl):
    points = points[:,:3]
    low = np.min(points, axis=0) - atom_size*2 - resl*2
    high = np.max(points, axis=0) + atom_size*2 + resl*2

    clashgrid = voxel_array.VoxelArray(low, high, np.array([resl]*3), bool)

    for pt in points:
        inds = clashgrid.indices_within_x_of(atom_size*2, pt)
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

@njit(fastmath=True)
def get_CB_from_xform(xform):
    CB_pos = np.array([0.52892494, -0.77445692, -1.19923854, 1.0], dtype=np.float32)
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

def set_ca_psi(points, tpose, psis, resno, psi):
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





# If unit vector is specified. Only points with positive dot products are kept
def prepare_context_by_dist_and_limits(context, pt, max_dist, unit_vector=None):
    pt = pt[:3]
    if ( not unit_vector is None ):
        vectors = context - pt
        dots = np.sum( np.multiply(vectors, unit_vector), axis=1)
        context = context[dots > 0]
    if ( len(context) == 0):
        context = np.array([[1000, 1000, 1000]])
    dists = np.linalg.norm( pt - context, axis=1 )
    context_dists = zip(context, dists)
    context_dists = sorted(context_dists, key=lambda x: x[1])
    context_by_dist, dists = zip(*context_dists)
    context_by_dist = np.array(context_by_dist)


    pos = 0
    context_dist_limits = []
    for dist in range(int(max_dist)+1):
        while ( pos < len(context_by_dist) and dists[pos] < dist ):
            pos += 1
        context_dist_limits.append(pos)

    context_dist_limits = np.array(context_dist_limits)

    return context_by_dist, context_dist_limits

# def clash_check_points_context(pts, point_dists, context_by_dist, context_dist_limits, clash_dist, max_clash):
#     if ( context_by_dist is None):
#         return 0
#     return jit_clash_check_points_context(pts, point_dists, context_by_dist, context_dist_limits, clash_dist, max_clash)

@njit(fastmath=True)
def clash_check_points_context(pts, point_dists, context_by_dist, context_dist_limits, clash_dist, max_clash, tol=0):
    clashes = 0
    clash_dist2 = clash_dist * clash_dist
    pts = pts[:,:3]
    for ipt in range(len(pts)):
        pt = pts[ipt]
        lo_limit = context_dist_limits[max(0, int(point_dists[ipt] - clash_dist - 1 - tol))]
        limit = context_dist_limits[int(point_dists[ipt] + clash_dist + tol)]
        context = context_by_dist[lo_limit:limit]
        clashes += np.sum( np.sum( np.square( pt - context ), axis=1 ) < clash_dist2 )

        if ( clashes >= max_clash ):
            return clashes
    return clashes



def xform_magnitude_sq( rts, lever2 ):

    trans_part = rts[...,:3,3]
    err_trans2 = np.sum(np.square(trans_part), axis=-1)

    rot_part = rts[...,:3,:3]
    traces = np.trace(rot_part,axis1=-1,axis2=-2)
    cos_theta = ( traces - 1 ) / 2

    # We clip to 0 here so that negative cos_theta gets lever as error
    clipped_cos = np.clip( cos_theta, 0, 1)

    err_rot2 = ( 1 - np.square(cos_theta) ) * lever2

    # err = np.sqrt( err_trans2 + err_rot2 )
    err =  err_trans2 + err_rot2 

    return err

def xform_magnitude( rts, lever2 ):

    return np.sqrt( xform_magnitude_sq( rts, lever2 ) )


# This would be better if it found the center of each cluster
# This requires nxn of each cluster though
def cluster_xforms( close_thresh, lever, xforms, inverse_xforms = None, info_every=None ):
    if ( inverse_xforms is None ):
        inverse_xforms = np.linalg.inv(xforms)

    size = len(xforms)
    min_distances = np.zeros(size, float)
    min_distances.fill(9e9)
    assignments = np.zeros(size, int)
    center_indices = []

    lever2 = lever*lever

    cur_index = 0

    while ( np.sqrt(min_distances.max()) > close_thresh ):

        distances = xform_magnitude_sq( inverse_xforms[cur_index] @ xforms, lever2 )

        changes = distances < min_distances
        assignments[changes] = cur_index
        min_distances[changes] = distances[changes]

        center_indices.append( cur_index )
        cur_index = min_distances.argmax()

        if ( not info_every is None ):
            if ( len(center_indices) % info_every == 0 ):
                print("Cluster round %i: max_dist: %6.3f   %8i"%(len(center_indices), np.sqrt(min_distances[cur_index]), cur_index))

    return center_indices, assignments


def center_of_mass( coords ):
    com = np.sum( coords, axis=-2 ) / coords.shape[-2]
    return com

def radius_of_gyration( coords, com=None):
    if (com is None):
        com = center_of_mass(coords)

    # The extra 1s will cancel here
    dist_from_com2 = np.square(np.sum( coords - com, axis=-1))

    return np.sqrt( np.sum(dist_from_com2) / coords.shape[-2] )

def xform_from_flat( twelve ):
    xform = np.identity(4)
    xform[:3,:3].flat = twelve[:9]
    xform[:3,3].flat = twelve[9:]
    return xform

def flat_from_xform( xform ):
    return list(xform[:3,:3].flat) + list(xform[:3,3].flat)


