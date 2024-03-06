#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <THC/THCAtomics.cuh>
#include <torch/extension.h>
#include <math.h>

using at::Half;
using at::Tensor;
using phalf = at::Half;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// define the threads in each block
#define THREADS_PER_BLOCK 512

// calculate the number of blocks acordding to the number of block and data length
inline int GET_BLOCKS(const int N, const int num_threads = THREADS_PER_BLOCK) {
  int optimal_block_num = (N + num_threads - 1) / num_threads;
  int max_block_num = 4096;
  return min(optimal_block_num, max_block_num);
}

#define MAXN 20
#define PI 3.1416
#ifdef INFINITY
/* INFINITY is supported */
#endif
/**** helper function and helper class ****/
#define cross(o,a,b) (a.x - o.x) * (b.y - o.y) - \
                     (a.y - o.y) * (b.x - o.x)

#define distance(a,b) (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)

#define xRotate(R, x, y) (x*R[0][0] + y*R[1][0])
#define yRotate(R, x, y) (x*R[0][1] + y*R[1][1])

/* Create a class to store pointsets information */
struct Point
{
  /* data */
  float x,y; // use to store loacation

  // builder
  __device__ Point() {};
  __device__ Point(float x, float y): x(x), y(y){}
};

template <typename T>
__device__ inline void jarvis(T * in_poly,
                              int & n_poly)
    {
    /**
     * Params:
     *      in_poly Point list : lengh -> 9
     *      n_poly [Tensor] (return) : as an input -> which present the number of point in the in_poly
     *    
     * Return:
     *      in_poly Point list: Used to store the convex point information within n_poly length
     *      n_poly int: it presents the number of convex points in poly, we can use this to iterate in_poly
    */


   // search the most left side point
   Point left = in_poly[0]; // init with the first element
   int left_idx = 0;
   for (int i = 0; i < n_poly; i ++)
   {
    if(left.x > in_poly[i].x || 
      (left.x == in_poly[i].x) && left.y < in_poly[i].y)
      {
        left = in_poly[i];    
        left_idx = i;
      }
   }

  // use cross to keep searing the convex point
  int start_idx = left_idx;
  int convex_num = 0;
  Point result[MAXN]; // this list is used to store convex hull for temporary

  while (true)
  {
    result[convex_num] = in_poly[start_idx]; // record convex point
    convex_num ++;
    int next_idx = left_idx;

    // keeping every point on the right left side of start_idx -> next_idx
    for (int i = 0; i < n_poly; i++)
    {
      float sign = cross(in_poly[start_idx], in_poly[i], in_poly[next_idx]);
      if (sign > 0 || 
         (sign == 0 && (distance(in_poly[start_idx], in_poly[i]) > distance(in_poly[start_idx], in_poly[next_idx]))))
         {
            next_idx = i;
         }
    }
    start_idx = next_idx;
    // if the convex hull is closed, then break the loop
    if (start_idx == left_idx)
    {
      break;
    }
  }

  // prepare for return value
  n_poly = convex_num;
  for (int i = 0; i < n_poly; i ++)
  {
    in_poly[i] = result[i];
  }
  }

template <typename T>
__device__ inline void findminRect(T const pts,
                                   const int convex_num,
                                   float *ret)
{
  float edge_list[MAXN];
  for (int i =0; i < convex_num; i ++)
  {
    float x_distance = pts[i].x  - pts[i + 1].x;
    float y_distance = pts[i].y  - pts[i + 1].y;

    edge_list[i] = atan2(y_distance, x_distance);
    
    // fixing angle
    if (edge_list[i] >= 0)
    {
      edge_list[i] = fmod(edge_list[i], PI/2);
    }else
    {
      edge_list[i] = edge_list[i] - (int(edge_list[i]/(PI/2) - 1)* (PI/2));
    }

    int unique_flag = 0;
    float unique_angle[MAXN];
    unique_angle[0] = edge_list[0];
    int n_unique = 1;
    int n_edge = convex_num;

    // remove duplicate edges
    for (int i = 0; i < n_edge; i ++)
    {
      for (int j = 0; j < n_unique; j ++)
      {
        if (edge_list[i] == unique_angle[i])
        {
          unique_flag ++;
        }
      }

      // if found unique angle
      if (unique_flag == 0)
      {
        unique_angle[n_unique] = edge_list[i];
        n_unique ++;
      }
      unique_flag = 0; // initialize unique flag
    }

    float min_area = INFINITY; // initalize minarea with minus infinity
    for (int i = 0; i < n_unique; i ++)
    {
      // build rotation matrix
      float R[2][2];
      R[0][0] = cos(unique_angle[i]);
      R[0][1] = -sin(unique_angle[i]);
      R[1][1] = cos(unique_angle[i]);
      R[1][0] = sin(unique_angle[i]);

      // build rotation point list
      Point rotated_point[MAXN];
      for (int i = 0; i < convex_num; i ++)
      {
        rotated_point[i].x = pts[i].x*R[0][0] + pts[i].y*R[1][0];
        rotated_point[i].y = pts[i].x*R[0][1] + pts[i].y*R[1][1];
      }

    float x_min = INFINITY;
    float y_min = INFINITY;
    float x_max = -1*INFINITY;
    float y_max = -1*INFINITY;

    for (int j = 0; j < convex_num; j ++)
    {
      if (rotated_point[j].x < x_min)
      {
        x_min = rotated_point[j].x;
      }

      if (rotated_point[j].y < y_min)
      {
        y_min = rotated_point[j].y;
      }

      if (rotated_point[j].x > x_max)
      {
        x_max = rotated_point[j].x;
      }

      if (rotated_point[j].y > y_max)
      {
        y_max = rotated_point[j].y;
      }
    }

    float area = (x_max - x_min) * (y_max - y_min);
    if(area <= min_area)
    {
      min_area = area;
      ret[0] = unique_angle[i];
      ret[1] = x_min;
      ret[2] = y_min;
      ret[3] = x_max;
      ret[4] = y_max;
    }
    }
  }
}

template <typename T>
__device__ inline void findminbox(T const *const pointset,
                                  T *minpoints){
    /**
     * Params:
     *      pointsets [Tensor] : [18] which presensts [x1, y1, x2, y2, ... , x9, y2]
     *      minpoints [Tensor] (return) : [8]
    */
    int n_convex = 9;
    Point ps1[MAXN]; // input of min area bouding box
    Point convex[MAXN]; // input of jarvis    

    // prepare for jarvis
    for (int i = 0; i < n_convex; i ++)
    {
      convex[i].x = pointset[2*i];
      convex[i].y = pointset[2*i + 1];
    }

    jarvis(convex, n_convex); // n_convex is the number of convex hull point, use it to iterate the convex list

    for (int i = 0; i < n_convex; i ++)
    {
      ps1[i].x = convex[i].x; 
      ps1[i].y = convex[i].y; 
    }

    // close the convex hull
    ps1[n_convex].x = convex[0].x;
    ps1[n_convex].y = convex[0].y;

    float ret[5] = {0}; // initialize with 0 
    int convex_num = n_convex + 1;
    findminRect(ps1, convex_num, ret);

    float angle = ret[0];
    float x_min = ret[1];
    float y_min = ret[2];
    float x_max = ret[3];
    float y_max = ret[4];
    float R[2][2];

    R[0][0] = cos(angle);
    R[1][0] = -sin(angle);
    R[1][1] = cos(angle);
    R[0][1] = sin(angle);

    minpoints[0] = xRotate(R,x_min,y_min);
    minpoints[1] = yRotate(R,x_min,y_min);
    minpoints[2] = xRotate(R,x_max,y_min);
    minpoints[3] = yRotate(R,x_max,y_min);
    minpoints[4] = xRotate(R,x_max,y_max);
    minpoints[5] = yRotate(R,x_max,y_max);
    minpoints[6] = xRotate(R,x_min,y_max);
    minpoints[7] = yRotate(R,x_min,y_max);
    }

template <typename T>
__global__ void min_area_rect_kernel(const int ex_n_boxes,
                                     const T* ex_boxes,
                                     T* minbox){
    /**
     * Params:
     *      num_pointsets [int]: number of points in pointset 
     *      pointsets [Tensor]: [N, 18]
     *      minbox(return) [Tensor]: [N, 8]
    */
   // using loop to calculate box for each 9 points
   CUDA_1D_KERNEL_LOOP(index, ex_n_boxes){
        const T *cur_box = ex_boxes + index*18; 
        T *cur_minbox = minbox + index * 8;
        findminbox(cur_box, cur_minbox);
   }
}

void launch_min_area_rect(const Tensor pointsets,
                                Tensor polygons
                         ) {
    // get input size
    int num_pointsets = pointsets.size(0);
    const int output_size= polygons.numel(); // return the total number of elements in polygons tensor
    min_area_rect_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(num_pointsets ,(float *)pointsets.data_ptr(), (float *)polygons.data_ptr());
}