#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernScanStep(int n, int d, const int* in, int* out) {
          int index = threadIdx.x + (blockIdx.x * blockDim.x);

          if (index >= n) {
            return;
          }

          if (index >= d) {
            out[index] = in[index] + in[index - d];
          }
          else {
            out[index] = in[index];
          }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_a;
            int* dev_b;

            cudaMalloc((void**)&dev_a, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));

            cudaMemcpy(dev_a, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int threadsPerBlock = 256;
            int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            for (int d = 1; d < ilog2ceil(n); d++) {
              // kernel for all k
              int shift = 1 << d; // Following slides '2^{d-1}'
              kernScanStep<<<blocks, threadsPerBlock >>>(n, shift, dev_a, dev_b);

              std::swap(dev_a, dev_b);
            }

            cudaFree(dev_a);
            cudaFree(dev_b);
            timer().endGpuTimer();
        }
    }
}
