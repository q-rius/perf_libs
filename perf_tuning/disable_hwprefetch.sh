sudo rdmsr -p 3 0x1a4 # core 3, msr 0x1a4 - handles hardware prefetch controls
sudo wrmsr -p 3 0x1a4 0x1 # sets the first bit 

1A4H 420 MSR_MISC_FEATURE_CONTROL Miscellaneous Feature Control (R/W)

0 Core L2 Hardware Prefetcher Disable (R/W)
If 1, disables the L2 hardware prefetcher, which fetches
additional lines of code or data into the L2 cache.

1 Core L2 Adjacent Cache Line Prefetcher Disable (R/W)
If 1, disables the adjacent cache line prefetcher, which
fetches the cache line that comprises a cache line pair
(128 bytes).

2 Core DCU Hardware Prefetcher Disable (R/W)
If 1, disables the L1 data cache prefetcher, which
fetches the next cache line into L1 data cache.

3 Core DCU IP Prefetcher Disable (R/W)
If 1, disables the L1 data cache IP prefetcher, which
uses sequential load history (based on instruction
pointer of previous loads) to determine whether to
prefetch additional lines.