#!/usr/bin/env python

import os
import sys
import math
import struct

from npose_util import *

from pyrosetta import *
from pyrosetta.rosetta import *


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

def xform_to_matrix(xform):
    mat = numeric.xyzMatrix_double_t()
    mat.xx = xform[_xx]
    mat.xy = xform[_xy]
    mat.xz = xform[_xz]
    mat.yx = xform[_yx]
    mat.yy = xform[_yy]
    mat.yz = xform[_yz]
    mat.zx = xform[_zx]
    mat.zy = xform[_zy]
    mat.zz = xform[_zz]

    return mat

def matrix_to_xform(mat):
    xform = np.identity(4)
    xform[_xx] = mat.xx
    xform[_xy] = mat.xy
    xform[_xz] = mat.xz
    xform[_yx] = mat.yx
    xform[_yy] = mat.yy
    xform[_yz] = mat.yz
    xform[_zx] = mat.zx
    xform[_zy] = mat.zy
    xform[_zz] = mat.zz

    return xform

def xform_to_vector(xform):
    vec = numeric.xyzVector_double_t()
    vec.x = xform[_x]
    vec.y = xform[_y]
    vec.z = xform[_z]

    return vec

def vector_to_xform(vec):
    xform = np.identity(4)
    xform[_x] = vec.x
    xform[_y] = vec.y
    xform[_z] = vec.z

    return xform


def to_vector(xyz):
    vec = numeric.xyzVector_double_t(xyz[0], xyz[1], xyz[2])
    return vec

def to_float_vector(xyz):
    vec = numeric.xyzVector_float_t(xyz[0], xyz[1], xyz[2])
    return vec

def from_vector(vec):
    xyz = np.array([0, 0, 0]).astype(float)
    xyz[0] = vec.x
    xyz[1] = vec.y
    xyz[2] = vec.z
    return xyz


# Stub of the last 3 heavy atoms  
def get_business_stub(res):
    nheavy = res.nheavyatoms()
    stub = get_stub_from_n_ca_c(from_vector(res.xyz(nheavy)), from_vector(res.xyz(nheavy-1)), from_vector(res.xyz(nheavy-2)))
    return stub


def get_stub_from_res(res):
    return get_stub_from_n_ca_c(from_vector(res.xyz("N")), from_vector(res.xyz("CA")), from_vector(res.xyz("C")))


def get_atoms_and_radii(pose, subset=None, atom_ids_too=False):

    atoms = []
    radii = []
    ids = []

    for seqpos in range(1, pose.size()+1):
        if ( not subset is None ):
            if ( not subset[seqpos] ):
                continue
        res = pose.residue(seqpos)
        for iatom in range(1, res.natoms()+1):
            atoms.append(from_vector(res.xyz(iatom)))
            radii.append(res.atom_type(iatom).lj_radius())
            ids.append((seqpos, iatom))

    if ( atom_ids_too ):
        return np.array(atoms), np.array(radii), np.array(ids)

    return np.array(atoms), np.array(radii)


def get_pose_cas(pose, subset=None):
    cas = []

    for seqpos in range(1, pose.size()+1):
        if ( not subset is None ):
            if ( not subset[seqpos] ):
                continue
        res = pose.residue(seqpos)

        cas.append(from_vector(res.xyz("CA")))

    return np.array(cas)




def get_sasa_surface(pose, probe_size=2.2, resl=0.5):

    atomic_depth = core.scoring.atomic_depth.AtomicDepth(pose, probe_size, False, resl, False)

    vertices = atomic_depth.get_surface_vertices()
    verts = np.zeros((len(vertices), 3))
    vert_normals = np.zeros((len(vertices), 3))

    for i in range(len(vertices)):
        vert = vertices[i]
        verts[i, 0] = vert.x
        verts[i, 1] = vert.y
        verts[i, 2] = vert.z
        vert_normals[i,0] = vert.pn.x
        vert_normals[i,1] = vert.pn.y
        vert_normals[i,2] = vert.pn.z


    faces = atomic_depth.get_surface_faces()
    face_centers = np.zeros((len(faces), 3))
    face_normals = np.zeros((len(faces), 3))
    face_areas = np.zeros(len(faces))

    for i in range(len(faces)):
        face = faces[i]
        face_centers[i] = (verts[face.a] + verts[face.b] + verts[face.c]) / 3
        face_normals[i,0] = face.pn.x
        face_normals[i,1] = face.pn.y
        face_normals[i,2] = face.pn.z
        face_areas[i] = face.area


    return verts, vert_normals, face_centers, face_normals, face_areas



def npose_from_pose(pose, return_npose_to_pose=False, allow_non_protein=False, allow_missing_atoms=False, dtype=np.float32):
    npose_to_pose = []
    for i in range(1, pose.size()+1):
        if ( allow_non_protein or pose.residue(i).is_protein() ):
            npose_to_pose.append(i)

    npose = np.zeros((len(npose_to_pose)*R, 4), dtype)
    npose.fill(np.nan)
    npose[:,3] = 1

    cb_fills = []

    for inpose, seqpos in enumerate(npose_to_pose):
        offset = inpose * R
        res = pose.residue(seqpos)
        for iat, atom_name in enumerate(ATOM_NAMES):
            if ( not res.has( atom_name ) ):

                if ( atom_name == "CB" ):
                    cb_fills.append(inpose)
                    continue
                else:
                    if ( not allow_missing_atoms ):
                        raise ValueError('Pose missing atom %s'%atom_name)

            npose[offset+iat,:3] = from_vector(res.xyz(atom_name))

    if ( len(cb_fills) > 0 ):
        cb_fills = np.array(cb_fills)
        ncac = npose.reshape(-1, R, 4)[cb_fills][:,[N, CA, C], :3]
        tpose = get_stubs_from_n_ca_c(ncac[:,0], ncac[:,1], ncac[:,2])
        cbs = build_CB(tpose)
        offsets = cb_fills*R+CB
        npose[offsets] = cbs

    if ( return_npose_to_pose ):
        return npose, npose_to_pose
    return npose












