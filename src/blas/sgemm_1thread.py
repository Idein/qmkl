# GPU accelerated single precision matrix multiplication (single thread)
import numpy as np
import struct
import time
import random
# from PIL import Image

from videocore.assembler import qpu, assemble, print_qbin
from videocore.driver import Driver

def mask(*idxs):
    values = [1]*16
    for idx in idxs:
        values[idx] = 0
    return values

@qpu
def sgemm_gpu_code(asm):
    B_CUR_IDX = 0
    K_IDX = 1
    I_IDX = 2;           NROWS_IDX = 2
    J_IDX = 3;           NCOLS_IDX = 3
    P_IDX = 4;           DMA_SETUP_IDX = 4
    Q_IDX = 5;           LOAD_BLOCKS_IDX = 5
    R_IDX = 6;           STORE_BLOCKS_IDX = 6
    A_CUR_IDX = 7
    C_CUR_IDX = 8
    A_BASE_IDX = 9
    B_BASE_IDX = 10
    C_BASE_IDX = 11
    A_STRIDE_IDX = 12
    B_STRIDE_IDX = 13
    C_STRIDE_IDX = 14
    COEF_ADDR_IDX = 15

    ra = [ ra0 , ra1 , ra2 , ra3 , ra4 , ra5 , ra6 , ra7,
           ra8 , ra9 , ra10, ra11, ra12, ra13, ra14, ra15,
           ra16, ra17, ra18, ra19, ra20, ra21, ra22, ra23,
           ra24, ra25, ra26, ra27, ra28, ra29, ra30, ra31 ]
    rb = [ rb0 , rb1 , rb2 , rb3 , rb4 , rb5 , rb6 , rb7,
           rb8 , rb9 , rb10, rb11, rb12, rb13, rb14, rb15,
           rb16, rb17, rb18, rb19, rb20, rb21, rb22, rb23,
           rb24, rb25, rb26, rb27, rb28, rb29, rb30, rb31 ]

    #==== Load constants ====
    # Load constants to r2.
    mov(r0, uniform)    # uniforms address
    mov(r2, 1)
    ldi(null, mask(P_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # p
    ldi(null, mask(Q_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # q
    ldi(null, mask(R_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # r
    ldi(null, mask(A_BASE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # Address of A[0,0]
    ldi(null, mask(B_BASE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # Address of B[0,0]
    ldi(null, mask(C_BASE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # Address of C[0,0]
    ldi(null, mask(A_STRIDE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # A stride
    ldi(null, mask(B_STRIDE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # B stride
    ldi(null, mask(C_STRIDE_IDX), set_flags=True)
    mov(r2, uniform, cond='zs')     # C stride
    ldi(null, mask(COEF_ADDR_IDX), set_flags=True)
    ldi(r1, 4*10)
    iadd(r2, r0, r1, cond='zs')     # address of alpha and beta

    #==== Variables ====

    # A_base = address of A[0,0] + (p+15)/16*16*A_stride
    # B_base = address of B[0,0]                         + (r+63)/64*64*4
    # C_base = address of C[0,0] + (p+15)/16*16*C_stride + (r+63)/64*64*4

    # A_cur = A_base - 16*i*A_stride
    # B_cur = B_base                 - 4*64*j
    # C_cur = C_base - 16*i*C_stride - 4*64*j

    rotate(broadcast, r2, -P_IDX)
    iadd(r0, r5, 15)
    shr(r0, r0, 4)
    shl(r0, r0, 4)                  # r0=(p+15)/16*16
    rotate(broadcast, r2, -R_IDX)
    ldi(r1, 63)
    iadd(r1, r1, r5)
    shr(r1, r1, 6)
    shl(r1, r1, 8)                  # r1=r/64*64*4
    rotate(broadcast, r2, -A_STRIDE_IDX)
    imul24(r3, r5, r0)              # r3=(p+15)/16*16*A_stride
    ldi(null, mask(A_BASE_IDX), set_flags=True)
    iadd(r2, r2, r3, cond='zs')
    ldi(null, mask(B_BASE_IDX), set_flags=True)
    iadd(r2, r2, r1, cond='zs')
    rotate(broadcast, r2, -C_STRIDE_IDX)
    imul24(r3, r5, r0)              # r3=(p+15)/16*16*C_stride
    ldi(null, mask(C_BASE_IDX), set_flags=True)
    iadd(r2, r2, r3, cond='zs', set_flags=False)
    iadd(r2, r2, r1, cond='zs')

    # Set stride for DMA to load and store C.
    rotate(broadcast, r2, -C_STRIDE_IDX)
    setup_dma_load_stride(r5)
    ldi(r1, 4*16)
    isub(r1, r5, r1)
    setup_dma_store_stride(r1)

    # Disable swapping of two TMUs.
    mov(tmu_noswap, 1)

    # Initialize column vectors.
    for i in range(32):
        mov(ra[i],  0.0).mov(rb[i],  0.0)

    #==== i-loop ====

    # Initialize i.
    # i=(p+15)/16.
    rotate(broadcast, r2, -P_IDX)
    iadd(r0, r5, 15)
    ldi(null, mask(I_IDX), set_flags=True)
    shr(r2, r0, 4, cond='zs')

    L.i_loop

    #==== j-loop ====

    # Initialize j.
    # j=(r+63)/64.
    rotate(broadcast, r2, -R_IDX)
    ldi(r0, 63)
    iadd(r0, r0, r5)
    ldi(null, mask(J_IDX), set_flags=True)
    shr(r2, r0, 6, cond='zs')

    rotate(broadcast, r2, -I_IDX)
    shl(r0, r5, 4)                          # r0=16*i
    rotate(broadcast, r2, -A_STRIDE_IDX)
    imul24(r0, r0, r5)                      # r0=16*i*A_stride
    rotate(broadcast, r2, -A_BASE_IDX)
    ldi(null, mask(A_CUR_IDX), set_flags=True)
    isub(r2, r5, r0, cond='zs')

    L.j_loop

    rotate(broadcast, r2, -I_IDX)
    shl(r0, r5, 4)                          # r0=16*i
    rotate(broadcast, r2, -C_STRIDE_IDX)
    imul24(r0, r0, r5)                      # r0=16*i*C_stride
    rotate(broadcast, r2, -J_IDX)
    shl(r1, r5, 8)                          # r1=4*64*j
    rotate(broadcast, r2, -C_BASE_IDX)
    ldi(null, mask(C_CUR_IDX), set_flags=True)
    isub(r2, r5, r0, cond='zs', set_flags=False)
    isub(r2, r2, r1, cond='zs')

    rotate(broadcast, r2, -B_BASE_IDX)
    ldi(null, mask(B_CUR_IDX), set_flags=True)
    isub(r2, r5, r1, cond='zs')

    # r1[e] = A_cur + A_stride*e   (e=element number)
    nop()
    rotate(broadcast, r2, -A_STRIDE_IDX)
    imul24(r0, element_number, r5)
    rotate(broadcast, r2, -A_CUR_IDX)
    iadd(r1, r0, r5)

    # Initialize loop delta.
    # r3[0] = B_stride
    # r3[1] = -1
    # r3[other] = 0

    mov(r3, 0)
    rotate(broadcast, r2, -B_STRIDE_IDX)
    ldi(null, mask(B_CUR_IDX), set_flags=True)
    mov(r3, r5, cond='zs')
    ldi(null, mask(K_IDX), set_flags=True)
    mov(r3, -1, cond='zs')

    #==== k-loop ====
    # r2[1] = q (k=q)
    nop()
    rotate(broadcast, r2, -Q_IDX)
    ldi(null, mask(K_IDX), set_flags=True)
    mov(r2, r5, cond='zs')

    mov(uniforms_address, r2)
    mov(tmu0_s, r1)
    iadd(r1, r1, 4)
    nop(sig='load tmu0')

    iadd(r2, r2, r3).mov(tmu0_s, r1)
    iadd(r1, r1, 4).fmul(r0, r4, uniform)
    fadd(ra0,  ra0,  r0).fmul(r0, r4, uniform)
    fadd(rb0,  rb0,  r0).fmul(r0, r4, uniform)

    wait_dma_store()

    L.k_loop

    for i in range(1, 31):
        fadd(ra[i],  ra[i],  r0).fmul(r0, r4, uniform)
        fadd(rb[i],  rb[i],  r0).fmul(r0, r4, uniform)
    fadd(ra31, ra31, r0).fmul(r0, r4, uniform)
    fadd(rb31, rb31, r0, sig='load tmu0').mov(uniforms_address, r2)
    iadd(r2, r2, r3).mov(tmu0_s, r1)
    jzc(L.k_loop)
    iadd(r1, r1, 4).fmul(r0, r4, uniform)      # delay slot
    fadd(ra0,  ra0,  r0).fmul(r0, r4, uniform) # delay slot
    fadd(rb0,  rb0,  r0).fmul(r0, r4, uniform) # delay slot

    #==== end of k-loop ====

    def setup_dma_load_block(block):
        # nrows = min(P-((P+15)/16-i)*16, 16)
        rotate(broadcast, r2, -P_IDX)
        iadd(r1, r5, 15)
        shr(r1, r1, 4)
        rotate(broadcast, r2, -I_IDX)
        isub(r1, r1, r5)
        shl(r1, r1, 4)
        rotate(broadcast, r2, -P_IDX)
        isub(r1, r5, r1)
        rotate(broadcast, r1, 0)
        ldi(r1, 16)
        imin(r1, r1, r5)
        band(r1, r1, 0xF)
        ldi(null, mask(NROWS_IDX), set_flags=True)
        mov(r3, r1, cond='zs')

        # ncols = min(R-(((R+63)/64-j)*4+block)*16, 16)
        ldi(r1, 63)
        rotate(broadcast, r2, -R_IDX)
        iadd(r1, r1, r5)
        shr(r1, r1, 6)
        rotate(broadcast, r2, -J_IDX)
        isub(r1, r1, r5)
        shl(r1, r1, 2)
        iadd(r1, r1, block)
        shl(r1, r1, 4)
        rotate(broadcast, r2, -R_IDX)
        isub(r1, r5, r1)
        rotate(broadcast, r1, 0)
        ldi(r1, 16)
        imin(r1, r1, r5)
        band(r1, r1, 0xF)
        ldi(null, mask(NCOLS_IDX), set_flags=True)
        mov(r3, r1, cond='zs')

        rotate(broadcast, r3, -NCOLS_IDX)
        shl(r1, r5, 4)  # ncols<<4
        rotate(broadcast, r3, -NROWS_IDX)
        bor(r1, r1, r5) # ncols<<4|nrows
        shl(r1, r1, 8)  # (ncols<<4|nrows)<<8
        shl(r1, r1, 8)  # (ncols<<4|nrows)<<8<<8 = ncols<<20|nrows<<16
        ldi(null, mask(DMA_SETUP_IDX), set_flags=True)
        mov(r3, r1, cond='zs')
        ldi(r1,
            0x80000000|    # setup_dma_load
            0<<28|         # 32bit
            0<<24|         # use the extended pitch setup register
            1<<12|         # vpitch=1
            0<<11|         # horizontal
            (block*16)<<4| # Y=16*block
            0)             # X=0
        rotate(broadcast, r3, -DMA_SETUP_IDX)
        bor(vpmvcd_rd_setup, r1, r5)

    def setup_dma_store_block(block):
        # nrows = min(P-((P+15)/16-i)*16, 16)
        rotate(broadcast, r2, -P_IDX)
        iadd(r1, r5, 15)
        shr(r1, r1, 4)
        rotate(broadcast, r2, -I_IDX)
        isub(r1, r1, r5)
        shl(r1, r1, 4)
        rotate(broadcast, r2, -P_IDX)
        isub(r1, r5, r1)
        rotate(broadcast, r1, 0)
        ldi(r1, 16)
        imin(r1, r1, r5)
        rotate(broadcast, r1, 0)
        ldi(r1, 0x1F)
        band(r1, r1, r5)
        ldi(null, mask(NROWS_IDX), set_flags=True)
        mov(r3, r1, cond='zs')

        # ncols = min(R-(((R+63)/64-j)*4+block)*16, 16)
        ldi(r1, 63)
        rotate(broadcast, r2, -R_IDX)
        iadd(r1, r1, r5)               # R+63
        shr(r1, r1, 6)                 # (R+63)/64
        rotate(broadcast, r2, -J_IDX)
        isub(r1, r1, r5)               # (R+63)/64-j
        shl(r1, r1, 2)                 # ((R+63)/64-j)*4
        iadd(r1, r1, block)            # ((R+63)/64-j)*4+block
        shl(r1, r1, 4)                 # (((R+63)/64-j)*4+block)*16
        rotate(broadcast, r2, -R_IDX)
        isub(r1, r5, r1)               # R-(((R+63)/64-j)*4+block)*16
        rotate(broadcast, r1, 0)
        ldi(r1, 16)
        imin(r1, r1, r5)
        rotate(broadcast, r1, 0)
        ldi(r1, 0x1F)
        band(r1, r1, r5)
        ldi(null, mask(NCOLS_IDX), set_flags=True)
        mov(r3, r1, cond='zs')

        # stride = C_stride - 4 * ncols
        imul24(r1, r1, 4)
        rotate(broadcast, r2, -C_STRIDE_IDX)
        isub(r1, r5, r1)
        setup_dma_store_stride(r1)

        rotate(broadcast, r3, -NROWS_IDX)
        shl(r1, r5, 7)  # nrows<<7
        rotate(broadcast, r3, -NCOLS_IDX)
        bor(r1, r1, r5) # nrows<<7|ncols
        shl(r1, r1, 8)  # (nrows<<7|ncols)<<8
        shl(r1, r1, 8)  # (nrows<<7|ncols)<<8<<8 = nrows<<23|ncols<<16
        ldi(null, mask(DMA_SETUP_IDX), set_flags=True)
        mov(r3, r1, cond='zs')
        ldi(r1,
            0x80000000|    # setup_dma_store
            1<<14|         # horizontal
            (16*block)<<7| # Y=16*block
            0<<3|          # X=0
            0)             # 32bit
        rotate(broadcast, r3, -DMA_SETUP_IDX)
        bor(vpmvcd_wr_setup, r1, r5)

    # blocks = min((R-((R+63)/64-j)*64+15)/16, 4)
    ldi(r1, 63)                   # 63
    rotate(broadcast, r2, -R_IDX)
    iadd(r1, r1, r5)              # R+63
    shr(r1, r1, 6)                # (R+63)/64
    rotate(broadcast, r2, -J_IDX)
    isub(r1, r1, r5)              # (R+63)/64-j
    shl(r1, r1, 6)                # ((R+63)/64-j)*64
    rotate(broadcast, r2, -R_IDX)
    isub(r1, r5, r1)              # R-((R+63)/64-j)*64
    iadd(r1, r1, 15)              # R-((R+63)/64-j)*64+15
    shr(r1, r1, 4)                # (R-((R+63)/64-j)*64+15)/16
    imin(r1, r1, 4)
    ldi(null, mask(LOAD_BLOCKS_IDX), set_flags=True)
    mov(r3, r1, cond='zs')
    ldi(null, mask(STORE_BLOCKS_IDX), set_flags=True)
    mov(r3, r1, cond='zs')

    # Issue load of block 0
    setup_dma_load_block(0)
    rotate(broadcast, r2, -C_CUR_IDX)
    start_dma_load(r5)
    rotate(broadcast, r3, -LOAD_BLOCKS_IDX)
    ldi(null, mask(LOAD_BLOCKS_IDX), set_flags=True)
    isub(r3, r5, 1, cond='zs')

    for i in range(1, 31):
        fadd(ra[i],  ra[i],  r0).fmul(r0, r4, uniform)
        fadd(rb[i],  rb[i],  r0).fmul(r0, r4, uniform)
    fadd(ra31, ra31, r0).fmul(r0, r4, uniform)
    fadd(rb31, rb31, r0, sig='load tmu0') # Emit load tmu0 signal for the last write to tmu0_s

    wait_dma_load() # block 0

    # Issue loading of block 1
    rotate(broadcast, r3, -LOAD_BLOCKS_IDX)
    mov(r0, r5, set_flags=True)
    jzs(L.skip_load_block_1)
    nop(); nop(); nop()
    setup_dma_load_block(1)
    ldi(r0, 4*16)
    rotate(broadcast, r2, -C_CUR_IDX)
    iadd(vpm_ld_addr, r5, r0)
    ldi(null, mask(LOAD_BLOCKS_IDX), set_flags=True)
    isub(r3, r3, 1, cond='zs')
    L.skip_load_block_1

    # Load alpha and beta.
    rotate(r0, r2, -COEF_ADDR_IDX)
    mov(uniforms_address, r0)

    # Setup VPM access for block 0
    setup_vpm_read(mode='32bit vertical', Y=0, X=0, nrows=16)
    setup_vpm_write(mode='32bit vertical', Y=0, X=0)

    mov(r1, uniform)        # r1=alpha
    mov(broadcast, uniform) # r5=beta

    fmul(ra0, ra0, r1)
    fmul(r0, vpm, r5)
    for i in range(7):
        fadd(vpm, ra[i], r0).fmul(rb[i], rb[i], r1)
        mov(ra[i], 0.0)     .fmul(r0, vpm, r5)
        fadd(vpm, rb[i], r0).fmul(ra[i+1], ra[i+1], r1)
        mov(rb[i], 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra7, r0).fmul(rb7, rb7, r1)
    mov(ra7, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb7, r0)
    mov(rb7, 0.0)

    wait_dma_load()

    # Issue store of block 0
    setup_dma_store_block(0)
    rotate(broadcast, r2, -C_CUR_IDX)
    start_dma_store(r5)
    ldi(null, mask(STORE_BLOCKS_IDX), set_flags=True)
    isub(r3, r3, 1, cond='zs')

    # Issue load of block 2.
    rotate(broadcast, r3, -LOAD_BLOCKS_IDX)
    mov(r0, r5, set_flags=True)
    jzs(L.skip_load_block_2)
    nop(); nop(); nop()
    setup_dma_load_block(2)
    ldi(r0, 4*16*2)
    rotate(broadcast, r2, -C_CUR_IDX)
    iadd(vpm_ld_addr, r5, r0)
    ldi(null, mask(LOAD_BLOCKS_IDX), set_flags=True)
    isub(r3, r3, 1, cond='zs')
    L.skip_load_block_2

    # Load alpha and beta.
    rotate(r0, r2, -COEF_ADDR_IDX)
    mov(uniforms_address, r0)

    # Setup VPM access for block 1
    setup_vpm_read(mode='32bit vertical', Y=16, X=0, nrows=16)
    setup_vpm_write(mode='32bit vertical', Y=16, X=0)

    mov(r1, uniform)        # r1=alpha
    mov(broadcast, uniform) # r5=beta

    fmul(ra8, ra8, r1)
    fmul(r0, vpm, r5)
    for i in range(8, 15):
        fadd(vpm, ra[i], r0).fmul(rb[i], rb[i], r1)
        mov(ra[i], 0.0)     .fmul(r0, vpm, r5)
        fadd(vpm, rb[i], r0).fmul(ra[i+1], ra[i+1], r1)
        mov(rb[i], 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra15, r0).fmul(rb15, rb15, r1)
    mov(ra15, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb15, r0)
    mov(rb15, 0.0)

    wait_dma_load()

    # Issue store of block 1
    rotate(broadcast, r3, -STORE_BLOCKS_IDX)
    mov(r0, r5, set_flags=True)
    jzs(L.skip_store_block_1)
    nop(); nop(); nop()
    setup_dma_store_block(1)
    ldi(r0, 4*16)
    rotate(broadcast, r2, -C_CUR_IDX)
    iadd(vpm_st_addr, r5, r0)
    ldi(null, mask(STORE_BLOCKS_IDX), set_flags=True)
    isub(r3, r3, 1, cond='zs')
    L.skip_store_block_1

    # Issue load of block 3
    rotate(broadcast, r3, -LOAD_BLOCKS_IDX)
    mov(r0, r5, set_flags=True)
    jzs(L.skip_load_block_3)
    nop(); nop(); nop()
    setup_dma_load_block(3)
    ldi(r0, 4*16*3)
    rotate(broadcast, r2, -C_CUR_IDX)
    iadd(vpm_ld_addr, r5, r0)
    ldi(null, mask(LOAD_BLOCKS_IDX), set_flags=True)
    isub(r3, r3, 1, cond='zs')
    L.skip_load_block_3

    # Load alpha and beta.
    rotate(r0, r2, -COEF_ADDR_IDX)
    mov(uniforms_address, r0)

    # setup VPM access for block 2.
    setup_vpm_read(mode='32bit vertical', X=0, Y=32, nrows=16)
    setup_vpm_write(mode='32bit vertical', X=0, Y=32)

    mov(r1, uniform)        # r1=alpha
    mov(broadcast, uniform) # r5=beta

    fmul(ra16, ra16, r1)
    fmul(r0, vpm, r5)
    for i in range(16, 23):
        fadd(vpm, ra[i], r0).fmul(rb[i], rb[i], r1)
        mov(ra[i], 0.0)     .fmul(r0, vpm, r5)
        fadd(vpm, rb[i], r0).fmul(ra[i+1], ra[i+1], r1)
        mov(rb[i], 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra23, r0).fmul(rb23, rb23, r1)
    mov(ra23, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb23, r0)
    mov(rb23, 0.0)

    wait_dma_load()

    # Issue store of block 2.
    rotate(broadcast, r3, -STORE_BLOCKS_IDX)
    mov(r0, r5, set_flags=True)
    jzs(L.skip_store_block_2)
    nop(); nop(); nop()
    setup_dma_store_block(2)
    ldi(r0, 4*16*2)
    rotate(broadcast, r2, -C_CUR_IDX)
    iadd(vpm_st_addr, r5, r0)
    ldi(null, mask(STORE_BLOCKS_IDX), set_flags=True)
    isub(r3, r3, 1, cond='zs')
    L.skip_store_block_2

    # Load alpha and beta.
    rotate(r0, r2, -COEF_ADDR_IDX)
    mov(uniforms_address, r0)

    # setup VPM access for block 3
    setup_vpm_read(mode='32bit vertical', X=0, Y=48, nrows=16)
    setup_vpm_write(mode='32bit vertical', X=0, Y=48)

    mov(r1, uniform)        # r1=alpha
    mov(broadcast, uniform) # r5=beta

    fmul(ra24, ra24, r1)
    fmul(r0, vpm, r5)
    for i in range(24, 31):
        fadd(vpm, ra[i], r0).fmul(rb[i], rb[i], r1)
        mov(ra[i], 0.0)     .fmul(r0, vpm, r5)
        fadd(vpm, rb[i], r0).fmul(ra[i+1], ra[i+1], r1)
        mov(rb[i], 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, ra31, r0).fmul(rb31, rb31, r1)
    mov(ra31, 0.0)     .fmul(r0, vpm, r5)
    fadd(vpm, rb31, r0)
    mov(rb31, 0.0)

    # Issue store of block 3
    rotate(broadcast, r3, -STORE_BLOCKS_IDX)
    mov(r0, r5, set_flags=True)
    jzs(L.skip_store_block_3)
    nop(); nop(); nop()
    setup_dma_store_block(3)
    ldi(r0, 4*16*3)
    rotate(broadcast, r2, -C_CUR_IDX)
    iadd(vpm_st_addr, r5, r0)
    ldi(null, mask(STORE_BLOCKS_IDX), set_flags=True)
    isub(r3, r3, 1, cond='zs')
    L.skip_store_block_3

    rotate(broadcast, r2, -J_IDX)
    isub(r0, r5, 1)
    jzc(L.j_loop)   # Jump iz Z-flags are clear
    ldi(null, mask(J_IDX), set_flags=True)  # delay slot
    mov(r2, r0, cond='zs')                  # delay slot
    nop()                                   # delay slot

    rotate(broadcast, r2, -I_IDX)
    isub(r0, r5, 1)
    jzc(L.i_loop)
    ldi(null, mask(I_IDX), set_flags=True)  # delay slot
    mov(r2, r0, cond='zs')                  # delay slot
    nop()                                   # delay slot

    # for i in range(16*1024):
    #     nop()

    wait_dma_store()

    exit()

def main():
    with Driver() as drv:
        p = random.randint(1, 1024)
        q = random.randint(2, 512)
        r = random.randint(1, 1024)

        assert(q >= 2)

        # Allocate matrices.
        C = drv.alloc((p, r), 'float32')
        A = drv.alloc((p, q), 'float32')
        B = drv.alloc((q, r), 'float32')

        # Initialize matrices.
        np.random.seed(0)
        alpha = 1.0
        beta = 1.0
        A[:] = np.random.randn(p, q) # np.ones(shape=(p, q)) #
        B[:] = np.random.randn(q, r) # np.ones(shape=(q, r)) #
        C[:] = np.random.randn(p, r) # np.ones(shape=(p, r)) # np.arange(p*r).reshape(p, r) + 1 #

        # Reference
        RA = A.copy()
        RB = B.copy()
        RC = C.copy()
        start = time.time()
        R = alpha*RA.dot(RB) + beta*RC
        elapsed_ref = time.time() - start

        # Allocate uniforms.
        uniforms = drv.alloc(12, 'uint32')
        uniforms[0] = uniforms.address
        uniforms[1] = p
        uniforms[2] = q
        uniforms[3] = r
        uniforms[4] = A.address
        uniforms[5] = B.address
        uniforms[6] = C.address
        uniforms[7] = A.strides[0]
        uniforms[8] = B.strides[0]
        uniforms[9] = C.strides[0]
        uniforms[10] = struct.unpack('L', struct.pack('f', alpha))[0]
        uniforms[11] = struct.unpack('L', struct.pack('f', beta))[0]

        # Allocate GPU program.
        code = drv.program(sgemm_gpu_code)

        # GPU
        start = time.time()
        drv.execute(
                n_threads=1,
                program=code,
                uniforms=uniforms
                )
        elapsed_gpu = time.time() - start

        # Image.fromarray(R.astype(np.uint8)).save("expected.png")
        # Image.fromarray(C.astype(np.uint8)).save("sgemm.png")

        # np.set_printoptions(threshold=np.inf)
        # print(C.astype(int))

        def Gflops(sec):
            return (2*p*q*r + 3*p*r)/sec * 1e-9

        print('==== sgemm example ({p}x{q} times {q}x{r}) ===='.format(
                p=p, q=q, r=r))
        print('threads: {}'.format(1))
        print('numpy: {:.4f} sec, {:.4f} Gflops'.format(
                elapsed_ref, Gflops(elapsed_ref)))
        print('GPU: {:.4f} sec, {:.4f} Gflops'.format(
                elapsed_gpu, Gflops(elapsed_gpu)))
        print('minimum absolute error: {:.4e}'.format(
                float(np.min(np.abs(R - C)))))
        print('maximum absolute error: {:.4e}'.format(
                float(np.max(np.abs(R - C)))))
        print('minimum relative error: {:.4e}'.format(
                float(np.min(np.abs((R - C) / R)))))
        print('maximum relative error: {:.4e}'.format(
                float(np.max(np.abs((R - C) / R)))))

if __name__ == '__main__':
    main()
    #print_qbin(sgemm_gpu_code)
