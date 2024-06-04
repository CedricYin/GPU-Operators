- sgemm0: naive
- sgemm1: using shared memory to cache global memory, each thread is responsible for a location of C
- sgemm2: using register to cache shared memory, each thread is responsible for RM*RN location of C
- sgemm3: using vector instruction (float4)
- sgemm4: double buffering
- sgemm5: remove shared memory bank conflic (but due to double buffering, still have some bank conflic. I do not know why now)
- sgemm6: without double buffering, and bank conflict free. but the performance slightly slower than sgemm5
  
## Test on T4 GPU

Peak: 
- fp32：8141 GFLOPS
- bandwidth：320.0 GB/s

|kernel|GFLOPS|% of cublas|
|-|-|-|
|naive|652.77|13.0304%|
|shared tiling|978.251|20.0119%|
|register tiling|4078.86|83.8988%|
|vector|4323.81|89.1279%|
|double buffering|4280.43|88.4402%|
|bank conflic|4981.63|106.268%|

