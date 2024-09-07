# GPU programming
## Brief Intro to GPU
### History
1. Shaders: OpenGL
2. Supercomputers: GPGPU, CUDA, OpenCL, 
### GPU Computing
Let's see how GPU computes step by step:  
1. Setup inputs on the **host**(CPU-accessible memory)
2. Allocate memory for outputs on the host CPU
3. Allocate memory for inputs on the GPU
4. Allocate memory for outputs on the GPU
5. Copy inputs from host to GPU (slow, using `cudaMemcpy`)
6. Start **GPU kernel** (function that executes on gpu – fast!)
7. Copy output from GPU to host (slow)

## Tools
* Compiler explorer is out best friend.

## Case Studies  
* Parallel Prefix Sum (prefix_sum.cu)

## References
* [caltech cs179 GPU computing](http://courses.cms.caltech.edu/cs179/)
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [cpp conference 2016](https://www.youtube.com/watch?v=KHa-OSrZPGo&t=1885s)
* [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
