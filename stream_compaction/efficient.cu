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

          int pad_n = 1 << ilog2ceil(n);

          int* dev_idata;
          int* dev_odata;
          int* dev_bools;
          int* dev_indices;

          cudaMalloc(&dev_idata, n * sizeof(int));
          cudaMalloc(&dev_odata, n * sizeof(int));
          cudaMalloc(&dev_bools, n * sizeof(int));
          cudaMalloc(&dev_indices, pad_n * sizeof(int));

          // Ensure that values beyond n are 0
          cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

          cudaMemset(dev_odata, 0, n * sizeof(int));
          cudaMemset(dev_bools, 0, n * sizeof(int));

          int threadsPerBlock = 256;
          int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
          int blocksPadded = (pad_n + threadsPerBlock - 1) / threadsPerBlock;

          timer().startGpuTimer();

          // Start by mapping input to boolean
          StreamCompaction::Common::kernMapToBoolean<<<blocks, threadsPerBlock>>>(n, dev_bools, dev_idata);

          cudaMemset(dev_indices, 0, pad_n * sizeof(int));
          cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

          // Scan over boolean array (not using scan() to avoid additional memory copies
          for (int d = 0; d < ilog2ceil(n); d++) {
            kernScanUpSweep << <blocksPadded, threadsPerBlock >> > (pad_n, d, dev_indices);
          }

          // Zero
          cudaMemset(dev_indices + pad_n - 1, 0,sizeof(int));

          for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
            kernScanDownSweep << <blocksPadded, threadsPerBlock >> > (pad_n, d, dev_indices);
          }

          // Scatter values (compact)
          StreamCompaction::Common::kernScatter<<<blocks, threadsPerBlock>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);

          timer().endGpuTimer();

          cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

          int last[2];
          cudaMemcpy(&last[0], dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
          cudaMemcpy(& last[1], dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

          cudaFree(dev_idata);
          cudaFree(dev_odata);
          cudaFree(dev_bools);
          cudaFree(dev_indices);

          // Count is : indices[n - 1] + bools[n - 1]
          return last[0] + last[1];
        }
    }
}
