CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Rayyan Shaik
  *  [LinkedIn](https://www.linkedin.com/in/rayyan-shaik)
* Tested on: Windows 11, AMD Ryzen 7 5700H @ 3.3GHz 32GB, Mobile RTX 3070 (Personal Computer)

### Features Implemented
- CPU-based exclusive scan and stream compaction
- "Naive" GPU exclusive scan
- "Work-Efficient" GPU scan and stream compaction

### Performance Analysis
#### Block Size vs Elapsed Time
The optimal block size was found to be `256`. This block size was used throughout the rest of the tests.

The average time was computed by averaging 10 runs of the "Work-Efficient" scan implementation.

| Block Size | Average Time |
| -------- | -------- |
| 32 | 27.87ms |
| 62 | 14.81ms | 
| 128 | 9.072 |
| 256 | 6.87ms |
| 512 | 6.969ms |
| 1024 | 9.0927 |

#### Exclusive Scan Performance vs Elapsed Time
This graph compares the average elapsed time of each scan implementation at specific array sizes. Each implementations time at an array size was the average timing of 10 trials.

![scan_performance.png](/Project2-Stream-Compaction/img/scan_performance.png)

| number of elements  | CPU (ms) | Naive (ms) | Work Efficient (Global) ms | Thrust    |
|---------------------|---------|----------|-------------------------|-----------|
| 2^8                 | 0.00012 | 0.3735  | 0.036               | 0.03402 |
| 2^10                | 0.0004  | 0.4389   | 0.059                  | 0.03583 |
| 2^12                | 0.0032 | 0.4366 | 0.069               | 0.07824 |
| 2^14                | 0.0083 | 0.5771 | 0.081               | 0.03569 |
| 2^16                | 0.0385 | 1.1286   | 0.158                | 0.07680   |
| 2^18                | 0.1889 | 0.9291 | 0.301                | 0.19146  |
| 2^20                | 0.5889 | 1.1315 | 0.567                | 0.18786   |
| 2^22                | 2.3937 | 3.3798  | 1.707                 | 0.25520   |
| 2^24                | 10.3350 | 10.2485  | 6.658                  | 0.51824   |
| 2^26                | 39.9080 | 45.3482  | 26.443                 | 1.65920   |
| 2^28                | 163.0230 | 190.081  | 112.529                 | 5.48830   |



Once measured, I found the the performance of _all_ GPU scan implementations to be somewhat weaker than I expected. For array sizes `<= 2^16`, the CPU scan was the best performing. With this data in mind, its likely due a combination of there being little memory overhead and simply having a "small" array of elements to work with. 

As the number of elements grow `>= 2^20`, we see that my GPU implementations (Naive, Work-Efficient) and the CPU implementation somewhat "converge" in efficiency. This was highly surprising as I expected the "Work-Efficient" scan to be a significant improvement over the CPU implementation, when it was in fact only faster at large inputs (`>= 2^22`) -- The Work Efficient implementation was ~31% faster than the CPU implemenation at 2^28 elements.

The Thrust `exclusive_scan` implemntation was by far the most effective, unsurprisingly. 

#### Thrust Exclusive Scan
![thrust_nsight.png](/Project2-Stream-Compaction/img/thrust_nsight.png)

Within this Nsight profiling screenshot, `thrust::exclusive_scan` executes for `[2.873 ms]`. It appears that Thrust allocates temporary device memory with cudaMalloc, launches the internal GPU work, synchronizes the CUDA stream, and then frees the temporary allocation with cudaFree. Most of this kernels duration is spent inside cudaStreamSynchronize, waiting on all threads to finish. It is notable that the internal memory allocation and deallocation takes `~410μs`, which is ~14% of the execution time.


#### Test Suite Output
`SIZE = 2^24`
`NPOT = SIZE - 3`
Block Size = `256`

```
****************
** SCAN TESTS **
****************
    [  25   6  23   3  37  12  46   2  35  32  47   6  14 ...  36   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 8.8536ms    (std::chrono Measured)
    [   0  25  31  54  57  94 106 152 154 189 221 268 274 ... 410838314 410838350 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 9.1714ms    (std::chrono Measured)
    [   0  25  31  54  57  94 106 152 154 189 221 268 274 ... 410838282 410838293 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 12.3334ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 9.76525ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 9.99114ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 6.54698ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.543808ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.555136ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   3   3   1   2   2   2   1   2   3   2   0 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 31.122ms    (std::chrono Measured)
    [   1   2   3   3   1   2   2   2   1   2   3   2   2 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 31.021ms    (std::chrono Measured)
    [   1   2   3   3   1   2   2   2   1   2   3   2   2 ...   1   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 57.6729ms    (std::chrono Measured)
    [   1   2   3   3   1   2   2   2   1   2   3   2   2 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 10.3637ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 11.537ms    (CUDA Measured)
    passed
```

