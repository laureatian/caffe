#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void vec_mul( 
          __global const float * A, 
          unsigned int A_row_start, unsigned int A_col_start, 
          unsigned int A_row_inc, unsigned int A_col_inc, 
          unsigned int A_row_size, unsigned int A_col_size, 
          unsigned int A_internal_rows, unsigned int A_internal_cols, 
          __global const float * v, 
          unsigned int v_start, unsigned int v_inc, unsigned int v_size, 
          __global float * result, 
          unsigned int result_start, unsigned int result_inc, unsigned int result_size, 
          __local float * work) 
{ 
  unsigned int row_gid = get_global_id(0) / get_local_size(0); 
  unsigned int col_gid = get_global_id(0) % get_local_size(0); 
  unsigned int lid = get_local_id(0); 
  for (unsigned int row = row_gid; row < A_row_size; row += get_num_groups(0)) 
  { 
    float4 dot_prod = 0;
    float4 a_temp = 0;
    float4 v_temp = 0; 
    for (unsigned int col = col_gid; col < A_col_size / 4; col+=get_local_size(0)){
      a_temp = vload4(col, A + row * A_internal_cols);
      v_temp = vload4(col, v); 
      dot_prod += a_temp * v_temp;
    } 
    work[lid] = dot(dot_prod, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
    for(unsigned int stride=get_local_size(0)/2 ; stride>0 ; stride>>=1){ 
      barrier(CLK_LOCAL_MEM_FENCE); 
      if(lid < stride) 
        work[lid] += work[lid+stride]; 
    } 
    if(lid == 0) 
      result[row * result_inc + result_start] = work[0]; 
  }
}

__kernel void vec_mul1( 
          __global const float * A, 
          unsigned int A_row_size, unsigned int A_col_size, 
          unsigned int x_pad, unsigned int y_pad,
          __global const float * v, 
          unsigned int v_size, 
          __global float * result, 
          unsigned int result_size, 
          __local float * work) 
{ 
  unsigned int row_gid = get_global_id(0) / get_local_size(0); 
  unsigned int col_gid = get_global_id(0) % get_local_size(0); 
  unsigned int lid = get_local_id(0); 
  for (unsigned int row = row_gid * y_pad; row < A_row_size && row < (row_gid+1) * y_pad; row += 1) 
  { 
    float4 dot_prod = 0;
    float4 a_temp = 0;
    float4 v_temp = 0; 
    //for (unsigned int col = col_gid * x_pad/4; col < A_col_size / 4 && col < (col_gid +1) * x_pad/4; col+=1){
    for (unsigned int col = col_gid; col < A_col_size /4; col+=get_local_size(0)){
      a_temp = vload4(col, A + row * A_col_size);
      v_temp = vload4(col, v); 
      dot_prod += a_temp * v_temp;
    } 
    work[lid] = dot(dot_prod, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
    for(unsigned int stride=get_local_size(0)/2 ; stride>0 ; stride>>=1){ 
      barrier(CLK_LOCAL_MEM_FENCE); 
      if(lid < stride) 
        work[lid] += work[lid+stride]; 
    } 
    if(lid == 0) 
      result[row] = work[0]; 
  }
}

__kernel void vec_mul2( 
          __global const float * A, 
          unsigned int A_row_size, unsigned int A_col_size, 
          __global const float * v, 
          unsigned int v_size, 
          __global float * result, 
          unsigned int result_size) 
{ 

  unsigned int row_gid = get_global_id(0);
  unsigned int gid = row_gid * A_col_size;
  float4 dot_prod = 0;
  float4 a_temp = 0;
  float4 v_temp = 0;
  //float* col_data = A + gid; 
  for (unsigned int col = 0; col < A_col_size / 4; col+=1) {
      a_temp = vload4(col, A + gid);
      v_temp = vload4(col, v); 
      dot_prod += a_temp * v_temp;
  } 
  result[row_gid] = dot(dot_prod, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 

}

#define VEC_SIZE 4
//4.56ms
__kernel void vec_mul3( 
          __global const float4 * A, 
          unsigned int A_row_size, unsigned int A_col_size, 
          __global const float4 * v, 
          unsigned int v_size, 
          __global float4 * result, 
          unsigned int result_size, 
          __local float4 * work) 
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

  unsigned i = lid;
  do
  {
    const float4 a0 = src0_read[i];
    const float4 a1 = src0_read[i + A_col_size];
    const float4 a2 = src0_read[i + 2 * A_col_size];
    const float4 a3 = src0_read[i + 3 * A_col_size];

    const float4 b0 = src1_read[i];
/*
    dot0 = mad(a0, (float4) b0.x, dot0);
    dot0 = mad(a0, (float4) b0.y, dot0);
    dot0 = mad(a0, (float4) b0.z, dot0);
    dot0 = mad(a0, (float4) b0.w, dot0);

    dot1 = mad(a1, (float4) b0.x, dot1);
    dot1 = mad(a1, (float4) b0.y, dot1);
    dot1 = mad(a1, (float4) b0.z, dot1);
    dot1 = mad(a1, (float4) b0.w, dot1);

    dot2 = mad(a2, (float4) b0.x, dot2);
    dot2 = mad(a2, (float4) b0.y, dot2);
    dot2 = mad(a2, (float4) b0.z, dot2);
    dot2 = mad(a2, (float4) b0.w, dot2);

    dot3 = mad(a3, (float4) b0.x, dot3);
    dot3 = mad(a3, (float4) b0.y, dot3);
    dot3 = mad(a3, (float4) b0.z, dot3);
    dot3 = mad(a3, (float4) b0.w, dot3);
*/
    dot0 += a0 * b0;
    dot1 += a1 * b0;
    dot2 += a2 * b0;
    dot3 += a3 * b0;
    i += get_local_size(0);  
  }
  while( i < A_col_size);
/*
  work[lid].x = dot(dot0, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
  work[lid].y = dot(dot1, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
  work[lid].z = dot(dot2, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
  work[lid].w = dot(dot3, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
*/
  work[lid].s0 = dot0.x + dot0.y + dot0.z + dot0.w;
  work[lid].s1 = dot1.x + dot1.y + dot1.z + dot1.w;
  work[lid].s2 = dot2.x + dot2.y + dot2.z + dot2.w;
  work[lid].s3 = dot3.x + dot3.y + dot3.z + dot3.w;
  for(unsigned int stride=get_local_size(0)/2 ; stride>0 ; stride>>=1) { 
      barrier(CLK_LOCAL_MEM_FENCE); 
      if(lid < stride) 
        work[lid] += work[lid+stride]; 
  }
  if(lid == 0)
    result[row_gid] = work[0];
}

//3.89ms
__kernel void vec_mul4( 
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
/*
    dot0 = mad(a0, (float4) b0.x, dot0);
    dot0 = mad(a0, (float4) b0.y, dot0);
    dot0 = mad(a0, (float4) b0.z, dot0);
    dot0 = mad(a0, (float4) b0.w, dot0);

    dot1 = mad(a1, (float4) b0.x, dot1);
    dot1 = mad(a1, (float4) b0.y, dot1);
    dot1 = mad(a1, (float4) b0.z, dot1);
    dot1 = mad(a1, (float4) b0.w, dot1);

    dot2 = mad(a2, (float4) b0.x, dot2);
    dot2 = mad(a2, (float4) b0.y, dot2);
    dot2 = mad(a2, (float4) b0.z, dot2);
    dot2 = mad(a2, (float4) b0.w, dot2);

    dot3 = mad(a3, (float4) b0.x, dot3);
    dot3 = mad(a3, (float4) b0.y, dot3);
    dot3 = mad(a3, (float4) b0.z, dot3);
    dot3 = mad(a3, (float4) b0.w, dot3);

    dot4 = mad(a4, (float4) b0.x, dot4);
    dot4 = mad(a4, (float4) b0.y, dot4);
    dot4 = mad(a4, (float4) b0.z, dot4);
    dot4 = mad(a4, (float4) b0.w, dot4);

    dot5 = mad(a5, (float4) b0.x, dot5);
    dot5 = mad(a5, (float4) b0.y, dot5);
    dot5 = mad(a5, (float4) b0.z, dot5);
    dot5 = mad(a5, (float4) b0.w, dot5);

    dot6 = mad(a6, (float4) b0.x, dot6);
    dot6 = mad(a6, (float4) b0.y, dot6);
    dot6 = mad(a6, (float4) b0.z, dot6);
    dot6 = mad(a6, (float4) b0.w, dot6);

    dot7 = mad(a7, (float4) b0.x, dot7);
    dot7 = mad(a7, (float4) b0.y, dot7);
    dot7 = mad(a7, (float4) b0.z, dot7);
    dot7 = mad(a7, (float4) b0.w, dot7);
*/
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
/*
  work[lid].s0 = dot(dot0, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
  work[lid].s1 = dot(dot1, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
  work[lid].s2 = dot(dot2, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
  work[lid].s3 = dot(dot3, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
  work[lid].s4 = dot(dot4, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
  work[lid].s5 = dot(dot5, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
  work[lid].s6 = dot(dot6, (float4)(1.0f, 1.0f, 1.0f, 1.0f)); 
  work[lid].s7 = dot(dot7, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
*/
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
