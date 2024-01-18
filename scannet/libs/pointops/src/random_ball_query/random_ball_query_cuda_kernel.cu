#include "../cuda_utils.h"
#include "random_ball_query_cuda_kernel.h"


namespace random_ball_query_utils{

template <typename DType>
__device__ void swap(DType *x, DType *y)
{
    DType tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ void reheap(float *dist, int *idx, int k)
{
    int root = 0;
    int child = root * 2 + 1;
    while (child < k)
    {
        if(child + 1 < k && dist[child+1] > dist[child])
            child++;
        if(dist[root] > dist[child])
            return;
        swap<float>(&dist[root], &dist[child]);
        swap<int>(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}


__device__ void heap_sort(float *dist, int *idx, int k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        swap<float>(&dist[0], &dist[i]);
        swap<int>(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}

__device__ int get_bt_idx(int idx, const int *offset)
{
    int i = 0;
    while (1)
    {
        if (idx < offset[i])
            break;
        else
            i++;
    }
    return i;
}
}  // namespace ball_query_utils

__global__ void random_ball_query_cuda_kernel(int m, int nsample,
                                              float min_radius, float max_radius, const int *__restrict__ order,
                                              const float *__restrict__ xyz, const float *__restrict__ new_xyz,
                                              const int *__restrict__ offset, const int *__restrict__ new_offset,
                                              int *__restrict__ idx, float *__restrict__ dist2) {
    // input: xyz (n, 3) new_xyz (m, 3)
    // output: idx (m, nsample) dist (m, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    new_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    dist2 += pt_idx * nsample;

    int bt_idx = random_ball_query_utils::get_bt_idx(pt_idx, new_offset);
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = offset[bt_idx - 1];
    int end = offset[bt_idx];

    float max_radius2 = max_radius * max_radius;
    float min_radius2 = min_radius * min_radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;

    for(int i = start; i < end; i++){
        float x = xyz[order[i] * 3 + 0];
        float y = xyz[order[i] * 3 + 1];
        float z = xyz[order[i] * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);

        if (d2 <= 1e-5 || (d2 >= min_radius2 && d2 < max_radius2)){
            dist2[cnt] = d2;
            idx[cnt] = order[i];
            cnt += 1;
            if (cnt >= nsample) break;
        }
    }

    if (cnt < nsample) {
        for (int i = cnt; i < nsample; i++){
            idx[i] = -1;
            dist2[i] = 1e10;
        }
    }
}

void random_ball_query_cuda_launcher(int m, int nsample,
                                     float min_radius, float max_radius, const int *order,
                                     const float *xyz, const float *new_xyz,
                                     const int *offset, const int *new_offset,
                                     int *idx, float *dist2) {
    // input: new_xyz: (m, 3), xyz: (n, 3), idx: (m, nsample)
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    random_ball_query_cuda_kernel<<<blocks, threads, 0>>>(m, nsample,
                                                          min_radius, max_radius, order,
                                                          xyz, new_xyz,
                                                          offset, new_offset,
                                                          idx, dist2);
}
