#kesav@qbuntu:~/workspace/nanoBench$ pwd
#/home/kesav/workspace/nanoBench
sudo ./nanoBench.sh -asm_init "MOV RAX, R14; SUB RAX, 8; MOV [RAX], RAX; MOV RBX, R14; SUB RBX,8; MOV [RBX], RBX;" -asm "MOV RAX, [RAX]; MOV RBX, [RBX]; ADD RDX, 1; ADD RCX, 1; ADD RSI, 1; ADD RBP, 1; ADD RDI, 1; ADD R8,1; ADD R9,1; ADD R10,1; ADD R11,1; ADD R12,1; ADD R13,1;" -config configs/cfg_AlderLakeP_common.txt -cpu 4 
CORE_CYCLES: 5.00
INST_RETIRED: 13.00
IDQ.MITE_UOPS: 13.00
IDQ.DSB_UOPS: 0.00
IDQ.MS_UOPS: 0.01
LSD.UOPS: 0.00
UOPS_ISSUED: 13.01
UOPS_EXECUTED: 4.18
UOPS_RETIRED.SLOTS: 13.01
UOPS_DISPATCHED_PORT.PORT_0: 0.34
UOPS_DISPATCHED_PORT.PORT_1: 0.38
UOPS_DISPATCHED_PORT.PORT_2_3_10: 2.00
UOPS_DISPATCHED_PORT.PORT_4_9: 0.00
UOPS_DISPATCHED_PORT.PORT_5_11: 1.00
UOPS_DISPATCHED_PORT.PORT_6: 0.47
UOPS_DISPATCHED_PORT.PORT_7_8: 0.00
BR_INST_RETIRED.ALL_BRANCHES: 0.00
BR_MISP_RETIRED.ALL_BRANCHES: 0.00
MEM_LOAD_RETIRED.L1_HIT: 2.00
MEM_LOAD_RETIRED.L1_MISS: 0.00
MEM_LOAD_RETIRED.L2_HIT: 0.00
MEM_LOAD_RETIRED.L2_MISS: 0.00
MEM_LOAD_RETIRED.L3_HIT: 0.00
MEM_LOAD_RETIRED.L3_MISS: 0.00

sudo ./nanoBench.sh -asm_init "MOV RAX, R14; SUB RAX, 8; MOV [RAX], RAX; MOV RBX, R14; SUB RBX,8; MOV [RBX], RBX;" -asm "MOV [RAX], RAX; MOV [RBX], RBX; ADD RDX, 1; ADD RCX, 1; ADD RSI, 1; ADD RBP, 1; ADD RDI, 1; ADD R8,1; ADD R9,1; ADD R10,1; ADD R11,1; ADD R12,1; ADD R13,1;" -config configs/cfg_AlderLakeP_common.txt -cpu 4 