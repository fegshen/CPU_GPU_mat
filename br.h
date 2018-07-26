/*
 * br.h
 *
 *  Created on: 2018年7月20日
 *      Author: shenf
 *      实现了基本CPU和GPU的向量类与矩阵类的数据结构
 *      为了方便处理以及运算的效率，不在类里面提供复杂的数值运算
 */

#ifndef BR_H_
#define BR_H_

#include <string.h>			/* memset, memcpy */
#include <stdlib.h>     	/* malloc, free */
#include <iostream>			/* swap */

#include "brkernel.h"

#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);



namespace BR{
//基本数据类型的重命名
typedef bool 				BOOL;
typedef char 				CHR;
typedef unsigned char 		UCHR;
typedef int 					INT;
typedef unsigned int 		UINT;
typedef float 				SP;
typedef double 				DP;
typedef unsigned long *	ULNG_p;
typedef SP * 				SP_p;
typedef DP * 				DP_p;

template <typename T> class BRDVec;
template <typename T> class BRDMat;

//CPU向量类
template <typename T>
class BRHVec{
private:
	INT nn;												//数组长度，基0
	T *v;													//指向数据
	template <typename U>
	friend void swap(BRHVec<U> &a,BRHVec<U> &b);	//交换数据
	friend class BRDVec<T>;
public:
	BRHVec():nn(0),v(nullptr){};						//默认构造函数
	explicit BRHVec(INT n);							//构造长度为n的向量
	BRHVec(const T &a, INT n);						//初始化为常量a
	BRHVec(const T *a, INT n);						//初始化为T类型数组a中的值
	BRHVec(const BRHVec &rhs);						//拷贝构造函数
	BRHVec(BRHVec &&rhs);								//移动构造函数
	explicit BRHVec(const BRDVec<T> &rhs);			//由GPU向量类构造CPU向量类
	BRHVec& operator=(const BRHVec &rhs);			//赋值运算符
	BRHVec& operator=(BRHVec &&rhs);				//移动赋值运算符
	BRHVec& operator=(const T &a);					//给每个元素赋值a
	inline T& operator[](const INT i) const{return v[i];}	//返回元素
	inline INT size() const{return nn;}				//返回长度
	inline T* getPointer(){return v;}				//获得指针
	~BRHVec();											//析构函数
};

//CPU矩阵类,行优先
template <typename T>
class BRHMat{
private:
	INT nn;												//行
	INT mm;												//列
	T *v;													//数据存储
	template <typename U>
	friend void swap(BRHMat<U> &a,BRHMat<U> &b);	//交换数据
	friend class BRDMat<T>;
public:
	BRHMat():nn(0),mm(0),v(nullptr){};				//默认构造函数
	BRHMat(INT n, INT m);								//构造n*m矩阵
	BRHMat(const T &a, INT n, INT m);				//初始化为常数值
	BRHMat(const T *a, INT n, INT m);				//初始化为数组a的元素
	BRHMat(const BRHMat &rhs);						//拷贝构造函数
	BRHMat(BRHMat &&rhs);								//移动构造函数
	explicit BRHMat(const BRDMat<T> &rhs);			//由GPU矩阵类构造CPU矩阵类
	BRHMat& operator=(const BRHMat &rhs);			//赋值运算符
	BRHMat& operator=(BRHMat &&rhs);				//移动赋值运算符
	BRHMat& operator=(const T &a);					//给每个元素赋值a
	inline T* operator[](const INT i) const;		//下标，指向行i
	inline INT nrows() const;							//返回行数
	inline INT ncols() const;							//返回列数
	inline T* getPointer(){return v;}				//获得指针
	~BRHMat();											//析构函数
};

//GPU向量类
template <typename T>
class BRDVec{
private:
	INT nn;												//数组长度，基0
	T *v;													//指向数据

	template <typename U>
	friend void swap(BRDVec<U> &a,BRDVec<U> &b);	//交换数据
	friend class BRHVec<T>;
public:
	//仅用于CPU函数中
	BRDVec():nn(0),v(nullptr){};						//默认构造函数
	explicit BRDVec(INT n);							//构造长度为n的向量
	BRDVec(const T &a, INT n);						//初始化为常量a
	BRDVec(const T *a, INT n);						//初始化为T类型GPU数组a中的值
	BRDVec(const BRDVec &rhs);						//拷贝构造函数
	BRDVec(BRDVec &&rhs);								//移动构造函数
	explicit BRDVec(BRHVec<T> &rhs);				//由CPU向量转GPU向量
	BRDVec& operator=(const T &a);					//给每个元素赋值a
	inline INT size() const{return nn;}				//返回长度
	inline T* getPointer(){return v;}				//获得指针
	~BRDVec();											//析构函数

	//仅用于GPU函数中
	__device__ T& operator[](const INT i) const{return v[i];}	//返回元素

	//下面的仅仅用于与CPU函数对应，尽量少用
	BRDVec& operator=(const BRDVec &rhs);			//赋值运算符
	BRDVec& operator=(BRDVec &&rhs);				//移动赋值运算符
};

//GPU矩阵类,行优先
template <typename T>
class BRDMat{
private:
	INT nn;												//行
	INT mm;												//列
	T *v;													//数据存储
	template <typename U>
	friend void swap(BRDMat<U> &a,BRDMat<U> &b);	//交换数据
	friend class BRHMat<T>;
public:
	//仅用于CPU
	BRDMat():nn(0),mm(0),v(nullptr){};				//默认构造函数
	BRDMat(INT n, INT m);								//构造n*m矩阵
	BRDMat(const T &a, INT n, INT m);				//初始化为常数值
	BRDMat(const T *a, INT n, INT m);				//初始化为数组a的元素
	BRDMat(const BRDMat &rhs);						//拷贝构造函数
	BRDMat(BRDMat &&rhs);								//移动构造函数
	explicit BRDMat(BRHMat<T> &rhs);				//由CPU矩阵转GPU矩阵
	BRDMat& operator=(const T &a);					//给每个元素赋值a
	inline INT nrows() const;							//返回行数
	inline INT ncols() const;							//返回列数
	inline T* getPointer(){return v;}				//获得指针
	~BRDMat();											//析构函数

	//仅用于GPU
	__device__ T* operator[](const INT i) const{return v+i*mm;};		//下标，指向行i

	//下面的仅仅用于与CPU函数对应，尽量少用
	BRDMat& operator=(const BRDMat &rhs);			//赋值运算符
	BRDMat& operator=(BRDMat &&rhs);				//移动赋值运算符
};

}

//CPU类的实现
namespace BR{
/*
 * CPU向量类的实现
 *
 * */
template <typename T>
void swap(BRHVec<T> &a,BRHVec<T> &b)
{
	std::swap(a.v,b.v);
	std::swap(a.nn,b.nn);
}

template <typename T>
BRHVec<T>::BRHVec(INT n)
{
	v=new T[n];
	nn=n;
}

template <typename T>
BRHVec<T>::BRHVec(const T &a, INT n):nn(n)
{
	v=new T[n];
	nn=n;
	for(int i=0;i!=nn;++i){
		v[i] = a;
	}
}

template <typename T>
BRHVec<T>::BRHVec(const T *a, INT n)
{
	v=new T[n];
	nn=n;
	memcpy(v,a,n*sizeof(T));
}

template <typename T>
BRHVec<T>::BRHVec(const BRHVec &rhs)
{
	v=new T[rhs.nn];
	nn=rhs.nn;
	memcpy(v,rhs.v,nn*sizeof(T));
}

template <typename T>
BRHVec<T>::BRHVec(BRHVec &&rhs)
{
	swap(*this,rhs);
}

template <typename T>
BRHVec<T>::BRHVec(const BRDVec<T> &rhs)
{
	v=new T[rhs.nn];
	nn=rhs.nn;
	cudaMemcpy(v,rhs.v,nn*sizeof(T),cudaMemcpyDeviceToHost);
}

template <typename T>
BRHVec<T>& BRHVec<T>::operator=(const BRHVec &rhs)
{
	//是否有更好的处理自赋值的方法？
	T* newp=new T[rhs.nn];
	delete []v;

	v=newp;
	nn=rhs.nn;
	memcpy(v,rhs.v,nn*sizeof(T));
	return *this;
}

template <typename T>
BRHVec<T>& BRHVec<T>::operator=(BRHVec &&rhs)
{
	swap(*this,rhs);
	return *this;
}

template <typename T>
BRHVec<T>& BRHVec<T>::operator=(const T &a)
{
	for(int i=0;i!=nn;++i){
		v[i]=a;
	}
	return *this;
}

template <typename T>
BRHVec<T>::~BRHVec()
{
	delete []v;
}

/*
 * CPU矩阵类的实现
 *
 * */
template <typename T>
void swap(BRHMat<T> &a,BRHMat<T> &b)
{
	std::swap(a.v,b.v);
	std::swap(a.nn,b.nn);
	std::swap(a.mm,b.mm);
}

template <typename T>
BRHMat<T>::BRHMat(INT n, INT m)
{
	v=new T[n*m];
	nn=n;
	mm=m;
}

template <typename T>
BRHMat<T>::BRHMat(const T &a, INT n, INT m)
{
	v=new T[n*m];
	nn=n;
	mm=m;

	int size=nn*mm;
	for(int i=0;i!=size;++i){
		v[i]=a;
	}
}

template <typename T>
BRHMat<T>::BRHMat(const T *a, INT n, INT m)
{
	v=new T[n*m];
	nn=n;
	mm=m;
	memcpy(v,a,nn*mm*sizeof(T));
}

template <typename T>
BRHMat<T>::BRHMat(const BRHMat &rhs)
{
	v=new T[rhs.nn*rhs.mm];
	nn=rhs.nn;
	mm=rhs.mm;
	memcpy(v,rhs.v,nn*mm*sizeof(T));
}

template <typename T>
BRHMat<T>::BRHMat(BRHMat &&rhs):v(nullptr)
{
	swap(*this,rhs);
}

template <typename T>
BRHMat<T>::BRHMat(const BRDMat<T> &rhs){
	v=new T[rhs.nn*rhs.mm];
	nn=rhs.nn;
	mm=rhs.mm;
	cudaMemcpy(v,rhs.v,nn*mm*sizeof(T),cudaMemcpyDeviceToHost);
}

template <typename T>
BRHMat<T>& BRHMat<T>::operator=(const BRHMat &rhs)
{
	T* newp=new T[rhs.nn*rhs.mm];
	delete []v;

	v=newp;
	nn=rhs.nn;
	mm=rhs.mm;
	memcpy(v,rhs.v,nn*mm*sizeof(T));
	return *this;
}

template <typename T>
BRHMat<T>& BRHMat<T>::operator=(BRHMat &&rhs)
{
	swap(*this,rhs);
	return *this;
}

template <typename T>
BRHMat<T>& BRHMat<T>::operator=(const T &a)
{
	int size=nn*mm;
	for(int i=0;i!=size;++i){
		v[i]=a;
	}
	return *this;
}

template <typename T>
T* BRHMat<T>::operator[](const INT i) const
{
	return v+i*mm;
}

template <typename T>
INT BRHMat<T>::nrows() const
{
	return nn;
}

template <typename T>
INT BRHMat<T>::ncols() const
{
	return mm;
}

template <typename T>
BRHMat<T>::~BRHMat()
{
	delete []v;
}


//GPU向量类的实现
template <typename T>
void swap(BRDVec<T> &a,BRDVec<T> &b)
{
	std::swap(a.v,b.v);
	std::swap(a.nn,b.nn);
}

template <typename T>
BRDVec<T>::BRDVec(INT n)
{
	checkCudaErrors(cudaMalloc(&v, n*sizeof(T)));
	nn=n;
}

template <typename T>
BRDVec<T>::BRDVec(const T &a, INT n)
{
	checkCudaErrors(cudaMalloc(&v, n*sizeof(T)));
	int blocksPerGrid = (n+BLOCK_SIZE-1)/BLOCK_SIZE;
	brdvecset<<<blocksPerGrid,BLOCK_SIZE>>>(v,a,n);
	nn=n;
}

template <typename T>
BRDVec<T>::BRDVec(const T *a, INT n)
{
	checkCudaErrors(cudaMalloc(&v, n*sizeof(T)));
	nn=n;
	cudaMemcpy(v,a,n*sizeof(T),cudaMemcpyDeviceToDevice);
}

template <typename T>
BRDVec<T>::BRDVec(const BRDVec &rhs)
{
	checkCudaErrors(cudaMalloc(&v, rhs.nn*sizeof(T)));
	nn=rhs.nn;
	cudaMemcpy(v,rhs.v,nn*sizeof(T),cudaMemcpyDeviceToDevice);
}

template <typename T>
BRDVec<T>::BRDVec(BRDVec &&rhs):v(nullptr)
{
	swap(*this,rhs);
}

template <typename T>
BRDVec<T>::BRDVec(BRHVec<T> &rhs)
{
	checkCudaErrors(cudaMalloc(&v, rhs.nn*sizeof(T)));
	nn=rhs.nn;
	cudaMemcpy(v,rhs.v,nn*sizeof(T),cudaMemcpyHostToDevice);
}

template <typename T>
BRDVec<T>& BRDVec<T>::operator=(const BRDVec &rhs)
{
	//是否有更好的处理自赋值的方法？
	if(rhs.nn==nn){
		cudaMemcpy(v,rhs.v,nn*sizeof(T),cudaMemcpyDeviceToDevice);
	}
	else{
		T* newp;
		cudaMalloc(&newp, rhs.nn*sizeof(T));
		cudaFree(v);

		v=newp;
		nn=rhs.nn;
		cudaMemcpy(v,rhs.v,nn*sizeof(T),cudaMemcpyDeviceToDevice);
	}

	return *this;
}

template <typename T>
BRDVec<T>& BRDVec<T>::operator=(BRDVec &&rhs)
{
	swap(*this,rhs);
	return *this;
}

template <typename T>
BRDVec<T>& BRDVec<T>::operator=(const T &a)
{
	int sizeGrid = (nn+BLOCK_SIZE-1)/BLOCK_SIZE;
	brdvecset<<<sizeGrid,BLOCK_SIZE>>>(v,a,nn);
	return *this;
}

template <typename T>
BRDVec<T>::~BRDVec()
{
	if(v!=nullptr){
		cudaFree(v);
	}
}

/*
 * GPU矩阵类的实现
 *
 * */
template <typename T>
void swap(BRDMat<T> &a,BRDMat<T> &b)
{
	std::swap(a.v,b.v);
	std::swap(a.nn,b.nn);
	std::swap(a.mm,b.mm);
}

template <typename T>
BRDMat<T>::BRDMat(INT n, INT m)
{
	checkCudaErrors(cudaMalloc(&v, n*m*sizeof(T)));
	nn=n;
	mm=m;
}

template <typename T>
BRDMat<T>::BRDMat(const T &a, INT n, INT m)
{
	checkCudaErrors(cudaMalloc(&v, n*m*sizeof(T)));
	nn=n;
	mm=m;

   // Invoke kernel,set mat
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid((m+BLOCK_SIZE-1) / dimBlock.x,
		   	   	   (n+BLOCK_SIZE-1) / dimBlock.y);
   brdmatset<<<dimGrid,dimBlock>>>(v,a,n,m);
}

template <typename T>
BRDMat<T>::BRDMat(const T *a, INT n, INT m)
{
	checkCudaErrors(cudaMalloc(&v, n*m*sizeof(T)));
	nn=n;
	mm=m;
	cudaMemcpy(v,a,n*m*sizeof(T),cudaMemcpyDeviceToDevice);
}

template <typename T>
BRDMat<T>::BRDMat(const BRDMat &rhs)
{
	checkCudaErrors(cudaMalloc(&v, rhs.nn*rhs.mm*sizeof(T)));
	nn=rhs.nn;
	mm=rhs.mm;
	cudaMemcpy(v,rhs.v,nn*mm*sizeof(T),cudaMemcpyDeviceToDevice);
}

template <typename T>
BRDMat<T>::BRDMat(BRDMat &&rhs):v(nullptr)
{
	swap(*this,rhs);
}

template <typename T>
BRDMat<T>& BRDMat<T>::operator=(const BRDMat &rhs)
{
	if(rhs.nn==nn && rhs.mm==mm){
		cudaMemcpy(v,rhs.v,nn*mm*sizeof(T),cudaMemcpyDeviceToDevice);
	}
	else{
		T* newp;
		cudaMalloc(&newp, rhs.nn*rhs.mm*sizeof(T));
		cudaFree(v);

		v=newp;
		nn=rhs.nn;
		mm=rhs.mm;
		cudaMemcpy(v,rhs.v,nn*mm*sizeof(T),cudaMemcpyDeviceToDevice);
	}

	return *this;
}

template <typename T>
BRDMat<T>::BRDMat(BRHMat<T> &rhs)
{
	checkCudaErrors(cudaMalloc(&v, rhs.nn*rhs.mm*sizeof(T)));
	nn=rhs.nn;
	mm=rhs.mm;
	cudaMemcpy(v,rhs.v,nn*mm*sizeof(T),cudaMemcpyHostToDevice);
}

template <typename T>
BRDMat<T>& BRDMat<T>::operator=(BRDMat &&rhs)
{
	swap(*this,rhs);
	return *this;
}

template <typename T>
BRDMat<T>& BRDMat<T>::operator=(const T &a)
{
   // Invoke kernel,set mat
   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   dim3 dimGrid((mm+BLOCK_SIZE-1) / dimBlock.x,
				   (nn+BLOCK_SIZE-1) / dimBlock.y);
   brdmatset<<<dimGrid,dimBlock>>>(v,a,nn,mm);
	return *this;
}


template <typename T>
INT BRDMat<T>::nrows() const
{
	return nn;
}

template <typename T>
INT BRDMat<T>::ncols() const
{
	return mm;
}

template <typename T>
BRDMat<T>::~BRDMat()
{
	if(v!=nullptr){
		cudaFree(v);
	}
}

}


#endif /* BR_H_ */
