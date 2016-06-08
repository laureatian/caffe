#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif
#define VEC_SIZE 4
__kernel void TEMPLATE(matvec_mul,Dtype)( 
          __global const float4 * A, 
          unsigned int A_row_size, unsigned int A_col_size, 
          __global const float4 * v, 
          unsigned int v_size, 
          __global float8 * result, 
          unsigned int result_size, 
          __local float8 * work) 
{
  A_col_size /= VEC_SIZE;
  v_size /= VEC_SIZE;
 
  unsigned int row_gid = get_group_id(0); 
  unsigned int lid = get_local_id(0);

  const __global float4 *src0_read = A + row_gid * A_col_size + lid;
  const __global float4 *src1_read = v + lid;
  float4 dot0 = (float4)(0.f);
  float4 dot1 = (float4)(0.f);
  float4 dot2 = (float4)(0.f);
  float4 dot3 = (float4)(0.f);
  float4 dot4 = (float4)(0.f);
  float4 dot5 = (float4)(0.f);
  float4 dot6 = (float4)(0.f);
  float4 dot7 = (float4)(0.f);

  unsigned i = lid;
  do
  {
    const float4 a0 = src0_read[i];
    const float4 a1 = src0_read[i + A_col_size];
    const float4 a2 = src0_read[i + 2 * A_col_size];
    const float4 a3 = src0_read[i + 3 * A_col_size];
    const float4 a4 = src0_read[i + 4 * A_col_size];
    const float4 a5 = src0_read[i + 5 *A_col_size];
    const float4 a6 = src0_read[i + 6 * A_col_size];
    const float4 a7 = src0_read[i + 7 * A_col_size];

    const float4 b0 = src1_read[i];

    dot0 += a0 * b0;
    dot1 += a1 * b0;
    dot2 += a2 * b0;
    dot3 += a3 * b0;
    dot4 += a4 * b0;
    dot5 += a5 * b0;
    dot6 += a6 * b0;
    dot7 += a7 * b0;
    i += get_local_size(0);  
  }
  while( i < A_col_size);

  work[lid].s0 = dot0.x + dot0.y + dot0.z + dot0.w;
  work[lid].s1 = dot1.x + dot1.y + dot1.z + dot1.w;
  work[lid].s2 = dot2.x + dot2.y + dot2.z + dot2.w;
  work[lid].s3 = dot3.x + dot3.y + dot3.z + dot3.w;
  work[lid].s4 = dot4.x + dot4.y + dot4.z + dot4.w;
  work[lid].s5 = dot5.x + dot5.y + dot5.z + dot5.w;
  work[lid].s6 = dot6.x + dot6.y + dot6.z + dot6.w;
  work[lid].s7 = dot7.x + dot7.y + dot7.z + dot7.w;

  for(unsigned int stride=get_local_size(0)/2 ; stride>0 ; stride>>=1) { 
      barrier(CLK_LOCAL_MEM_FENCE); 
      if(lid < stride) 
        work[lid] += work[lid+stride]; 
  }

  if(lid == 0)
    result[row_gid] = work[0];
}
