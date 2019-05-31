#!/usr/bin/env python

import os
import sys
import itertools
import numpy as np

from numba import njit

# OOB gets clipped to the edges. Be careful to leave them at 0
class VoxelArray:

    def __init__(self, lbs, ubs, cbs, dtype="f8"):

        self.dim = len(lbs)
        self.lb = lbs
        self.ub = ubs
        self.cs = cbs

        extents = self.floats_to_indices_no_clip(np.array([self.ub]))[0]
        extents += 1
        self.arr = np.zeros(extents).astype(dtype)


    def floats_to_indices_no_clip(self, pts):
        inds = np.zeros((len(pts), self.dim)).astype(int)
        for i in range(self.dim):
            inds[:,i] = ((pts[:,i] - self.lb[i] ) / self.cs[i]).astype(int)
        return inds

    def floats_to_indices(self, pts):
        inds = np.zeros((len(pts), self.dim)).astype(int)
        for i in range(self.dim):
            inds[:,i] = np.clip( (pts[:,i] - self.lb[i] ) / self.cs[i], 0, self.arr.shape[i]-1).astype(int)
        return inds

    def indices_to_centers(self, inds ):
        pts = np.zeros((len(inds), self.dim))
        for i in range(self.dim):
            pts[:,i] = (inds[:,i] + 0.5)*self.cs[i] + self.lb[i]
        return pts

    def all_indices(self):
        ranges = []
        for i in range(self.dim):
            ranges.append(list(range(self.arr.shape[i])))
        inds = np.array(list(itertools.product(*ranges)))
        return inds

    def all_centers(self):
        inds = self.all_indices()
        return self.indices_to_centers(inds)


    # One would usuallly type assert(voxel.oob_is_zero())
    def oob_is_zero(self):
        # This could certainly be made more efficient
        all_indices = self.all_indices()
        is_good = np.zeros(len(all_indices))
        for i in range(self.dim):
            is_good |= (all_indices[:,i] == 0) | (all_indices[:,i] == self.arr.shape[i]-1)

        good_indices = all_indices[is_good]
        return np.any(self.arr[good_indices])

    # This uses the centers as measurement
    def indices_within_x_of(self, _x, pt):
        low = pt - _x
        high = pt + _x

        # If you hit these, you are about to make a mistake
        assert( not np.any( low <= self.lb + self.cs))
        assert( not np.any( high >= self.ub - self.cs ) )

        bounds = self.floats_to_indices( np.array( [low, high] ) )

        ranges = []
        for i in range(self.dim):
            ranges.append(np.arange(bounds[0, i], bounds[1, i] + 1) )

        
        # indices = np.array(itertools.product(*ranges))
        indices = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(ranges))

        centers = self.indices_to_centers(indices)

        return indices[ np.linalg.norm(centers - pt, axis=1) < _x ]




    def dump_grids_true(self, fname, func):
        centers = self.all_centers()
        vals = self.arr[floats_to_indices(centers)]

        f = open(fname, "w")

        anum = 1
        rnum = 1

        for ind, xyz in enumerate(centers):
            val = vals[ind]
            if (not func(val)):
                continue

            f.write("%s%5i %4s %3s %s%4i    %8.3f%8.3f%8.3f%6.2f%6.2f %11s\n"%(
                "HETATM",
                anum,
                "VOXL",
                "VOX",
                "A",
                rnum,
                xyz[0],xyz[1],xyz[2],
                1.0,
                1.0,
                "HB"
                ))

            anum += 1
            rnum += 1
            rnum %= 10000

        f.close()



    def clash_check(self, pts, max_clashes):
        assert(self.dim == 3)

        return numba_clash_check(pts, max_clashes, self.arr, self.lb, self.cs)



@njit(fastmath=True)
def xform_1_pt(pt, lb, cs, shape):
    x = np.int( ( pt - lb ) / cs )
    if ( x <= 0 ):
        return np.int(0)
    if ( x >= shape-1 ):
        return shape-1
    return x

@njit(fastmath=True)
def numba_clash_check(pts, max_clashes, arr, lb, cs):
    
    clashes = 0

    for i in range(len(pts)):
        pt = pts[i]
        x = xform_1_pt(pt[0], lb[0], cs[0], arr.shape[0])
        y = xform_1_pt(pt[1], lb[1], cs[1], arr.shape[1])
        z = xform_1_pt(pt[2], lb[2], cs[2], arr.shape[2])

        clashes += arr[x, y, z]

        if ( clashes > max_clashes ):
            return clashes

    return clashes















