#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int prefix = 0;
            for (int i = 0; i < n; i++) {
              odata[i] = prefix;
              prefix += idata[i];
            }
            timer().endCpuTimer();
        }

        // Exact same as scan() , except it does not start a timer.
        // Exists soley for timing purposes within compactWithScan()
        void scanUntimed(int n, int* odata, const int* idata) {
          int prefix = 0;
          for (int i = 0; i < n; i++) {
            odata[i] = prefix;
            prefix += idata[i];
          }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int head = 0;
            for (int i = 0; i < n; i++) {
              if (idata[i] != 0) {
                odata[head] = idata[i];
                head++;
              }
            }
            timer().endCpuTimer();
            return head;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* temp = new int[n];
            int* scanned = new int[n];
            int count = 0;

            for (int i = 0; i < n; i++) {
              temp[i] = (idata[i] != 0) ? 1 : 0;
            }

            scanUntimed(n, scanned, temp);

            for (int i = 0; i < n; i++) {
              if (temp[i] == 1) {
                odata[scanned[i]] = idata[i];
                count++;
              }
            }
            timer().endCpuTimer();
            return count;
        }
    }
}
