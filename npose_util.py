#!/usr/bin/env python

import os
import sys

import pandas as pd
import numpy as np
import warnings

N = 0
CA = 1
C = 2
O = 3
CB = 4
R = 5

def readpdb(fname):
    n = 'het ai an rn ch ri x y z occ bfac elem'.split()
    w = (6, 5, 5, 4, 2, 4, 12, 8, 8, 6, 6, 99)
    assert len(n) is len(w)
    compression = "gzip" if fname.endswith(".gz") else None
    df = pd.read_fwf(fname, widths=w, names=n, compression=compression)
    df = df[np.logical_or(df.het == 'ATOM', df.het == 'HETATM')]
    df.het = df.het == 'HETATM'
    df.ai = df.ai.astype('i4')
    # df.an = df.an.astype('S4')  # f*ck you, pandas!
    # df.rn = df.rn.astype('S3')  # f*ck you, pandas!
    # df.ch = df.ch.astype('S1')  # f*ck you, pandas!
    df.ri = df.ri.astype('i4')
    df.x = df.x.astype('f4')
    df.y = df.y.astype('f4')
    df.z = df.z.astype('f4')
    df.occ = df.occ.astype('f4')
    df.bfac = df.bfac.astype('f4')
    # df.elem = df.elem.astype('S4')  # f*ck you, pandas!
    return df


# N CA C O CB
# Assumes no glycines
warnings.filterwarnings("ignore", 'This pattern has match groups')
def npose_from_file(fname):
    pdpose = readpdb(fname)

    just_my_atoms = pdpose[pdpose.an.str.contains("^(N|CA|C|O|CB)$")]

    nres = len(just_my_atoms.ri.unique())
    # Could theoretically place a CB if it's missing
    assert( len(just_my_atoms) / nres == R )

    npose = just_my_atoms[['x', 'y', 'z', 'z']].values.astype('f8')
    npose[:,3] = 1.0

    return npose

def nsize(npose):
    return int(len(npose)/R)

def get_res( npose, resnum):
    return npose[R*resnum:R*(resnum+1)]


def get_stub_from_npose(npose, resnum):
    # core::kinematics::Stub( CA, N, C )

    res = get_res(npose, resnum)

    e1 = res[CA][:3] - res[N][:3]
    e1 /= np.linalg.norm(e1)

    e3 = np.cross( e1, res[C][:3] - res[N][:3] )
    e3 /= np.linalg.norm(e3)

    e2 = np.cross( e3, e1 )

    stub = np.zeros((4, 4))
    stub[...,:3,0] = e1
    stub[...,:3,1] = e2
    stub[...,:3,2] = e3
    stub[...,:3,3] = res[CA][:3]
    stub[...,3,3] = 1.0

    return stub

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

_atom_names = [" N  ", " CA ", " C  ", " O  ", " CB "]

def dump_npdb(npose, fname):
    with open(fname, 'w') as out:
        for i, a in enumerate(npose):
            out.write( format_atom(
                atomi=i+1,
                resn='ALA',
                resi=int(i/R)+1,
                atomn=_atom_names[i%R],
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
