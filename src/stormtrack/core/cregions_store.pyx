# !/usr/bin/env python3 
from __future__ import print_function

# C: C libraries
from libc.stdlib cimport free
from libc.stdlib cimport malloc

# C: Third-party
cimport cython
cimport numpy as np

# Standard library
import logging as log

# Third-party
import numpy as np


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_create_cregion
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::cregions_store::cregions_store_extend
cdef cRegion* cregions_store_get_new_region(cRegionsStore* store):

    # Assess current situation
    cdef bint no_blocks = (store.n_blocks == 0)
    cdef bint in_last_block = (store.i_block + 1 == store.n_blocks)
    cdef bint at_end_of_block = (store.i_next_region == store.block_size)
    cdef bint blocks_full = (in_last_block and at_end_of_block)

    # Allocate new block if necessary
    if no_blocks or blocks_full:
        cregions_store_extend(store)
        no_blocks = False
        in_last_block = True
        at_end_of_block = False
        blocks_full = False

    # Pick next available region
    cdef cRegion* cregion
    if not at_end_of_block:
        cregion = &store.blocks[store.i_block][store.i_next_region]
        store.i_next_region += 1
    elif not in_last_block:
        store.i_block += 1
        cregion = &store.blocks[store.i_block][0]
        store.i_next_region = 1
    else:
        _name_ = "cregions_store_get_new_region"
        log.error(f"{_name_}: should not happen")
        # SR_DBG <
        log.debug("")
        log.debug("no_blocks {no_blocks}")
        log.debug("in_last_block {in_last_block}")
        log.debug("at_end_of_block {at_end_of_block}")
        log.debug("blocks_full {blocks_full}")
        # SR_DBG >
        exit(4)
    return cregion


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_reset
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::cregion::cregion_reset
cdef void cregions_store_reset(cRegionsStore* store):
    cdef int i
    cdef int j
    for i in range(store.n_blocks):
        for j in range(store.block_size):
            cregion_reset(
                &store.blocks[i][j], unlink_pixels=True, reset_connected=True,
            )
    store.i_block = 0
    store.i_next_region = 0


# :call: > --- callers ---
# :call: > stormtrack::core::cregions_store::cregions_store_get_new_region
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegion
# :call: v stormtrack::core::structs::cRegionConf
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::structs::cregion_conf_default
# :call: v stormtrack::core::cregion::cregion_get_unique_id
# :call: v stormtrack::core::cregion::cregion_init
cdef void cregions_store_extend(cRegionsStore* store):
    cdef int i
    cdef int nold = store.n_blocks
    cdef int nnew = nold + 1
    cdef cRegion** blocks_tmp
    # print("< cregions_store_extend {nold} -> {nnew}")
    if nold > 0:
        blocks_tmp = <cRegion**>malloc(nold*sizeof(cRegion*))
        for i in range(nold):
            blocks_tmp[i] = store.blocks[i]

    if store.blocks is not NULL:
        free(store.blocks)
    store.blocks = <cRegion**>malloc(nnew*sizeof(cRegion*))

    if nold > 0:
        for i in range(nold):
            store.blocks[i] = blocks_tmp[i]
        free(blocks_tmp)

    store.blocks[nold] = <cRegion*>malloc(store.block_size*sizeof(cRegion))
    cdef cRegionConf conf = cregion_conf_default()
    for i in range(store.block_size):
        cregion_init(&store.blocks[nold][i], conf, cregion_get_unique_id())

    store.i_block = nnew - 1
    store.n_blocks = nnew
    store.i_next_region = 0


# :call: > --- callers ---
# :call: > stormtrack::core::grid::grid_cleanup
# :call: v --- calling ---
# :call: v stormtrack::core::structs::cRegionsStore
# :call: v stormtrack::core::cregion::cregion_cleanup
cdef void cregions_store_cleanup(cRegionsStore* store):
    cdef int i
    if store.blocks is not NULL:
        for i in range(store.n_blocks):
            for j in range(store.block_size):
                cregion_cleanup(
                    &store.blocks[i][j], unlink_pixels=False, reset_connected=False,
                )
            free(store.blocks[i])
            store.blocks[i] = NULL
        free(store.blocks)
        store.blocks = NULL
    store.i_block = 0
    store.n_blocks = 0
    store.i_next_region = 0
