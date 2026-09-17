#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernScanUpSweep(int n, int d, int* data) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          int k = index * (1 << (d + 1));

          if (k >= n) {
            return;
          }

          int left = k + (1 << d) - 1;
          int right = k + (1 << (d + 1)) - 1;

          if (right < n) {
            data[right] += data[left];
          }
        }

        __global__ void kernScanDownSweep(int n, int d, int* data) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);
          int k = index * (1 << (d + 1));

          if (k >= n) {
            return;
          }
          
          int left = k + (1 << d) - 1;
          int right = k + (1 << (d + 1)) - 1;

          if (right < n) {
            int temp = data[k + (1 << d) - 1];
            data[left] = data[right];
            data[right] += temp;
          }
   
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
          int pad_n = 1 << ilog2ceil(n); // Next power of 2 from n
          int* dev_data;

          cudaMalloc(&dev_data, pad_n * sizeof(int));

          // Ensure that values beyond n are 0
          cudaMemset(dev_data, 0, pad_n * sizeof(int));
          cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
          
          int threadsPerBlock = 256;
          int blocks = (pad_n + threadsPerBlock - 1) / threadsPerBlock;

          timer().startGpuTimer();
          for (int d = 0; d < ilog2ceil(n); d++) {
            // kernel for all k
            kernScanUpSweep << <blocks, threadsPerBlock >> > (pad_n, d, dev_data);
          }

          // Set data[n-1] <- 0
          cudaMemset(dev_data + pad_n - 1, 0, sizeof(int));

          for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
            // kernel for all k
            kernScanDownSweep << <blocks, threadsPerBlock >> > (pad_n, d, dev_data);
          }

          timer().endGpuTimer();

          cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
          cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
          timer().startGpuTimer();
            
            timer().endGpuTimer();
            return -1;
        }
    }
}
