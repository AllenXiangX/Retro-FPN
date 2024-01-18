#ifndef _RANDOM_BALL_QUERY_CUDA_KERNEL
#define _RANDOM_BALL_QUERY_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void random_ball_query_cuda(int m, int nsample,
                            float min_radius, float max_radius, at::Tensor order_tensor,
                            at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                            at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                            at::Tensor idx_tensor, at::Tensor dist2_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void random_ball_query_cuda_launcher(int m, int nsample,
                                    float min_radius, float max_radius,  const int *order,
                                    const float *xyz, const float *new_xyz,
                                    const int *offset, const int *new_offset,
                                    int *idx, float *dist2);

#ifdef __cplusplus
}
#endif
#endif
