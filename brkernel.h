/*
 * brkernel.h
 *
 *  Created on: 2018年7月25日
 *      Author: shenf
 */

#ifndef BRKERNEL_H_
#define BRKERNEL_H_

//常用的宏
#define BLOCK_SIZE 32


//设置向量的元素全为a
template <typename T>
__global__ void brdvecset(T *v,const T a,int N)
{
	int i=blockDim.x * blockIdx.x+threadIdx.x;
	if(i<N)
		v[i]=a;
}

//设置矩阵的元素全为a
template <typename T>
__global__ void brdmatset(T *v,const T a,int N,int M)
{
	int col=blockDim.x * blockIdx.x+threadIdx.x;
	int row=blockDim.y * blockIdx.y+threadIdx.y;
	if(row<N && col<M){
		int index=row*M+col;
		v[index]=a;
	}
}

#endif /* BRKERNEL_H_ */
