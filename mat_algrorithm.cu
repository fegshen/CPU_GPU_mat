#include <stdio.h>
#include "br.h"


class A
{
private:
	BR::BRHMat<BR::SP> a;
	BR::BRHMat<BR::SP> c;
public:
	explicit A(BR::BRHMat<BR::SP> &b):c(b)
	{
		a=BR::BRHMat<BR::SP>(b);
		printf("%f\n",a[1][0]);
		printf("%f",c[2][0]);
	}
};

int main()
{
	BR::BRDVec<BR::SP> a(4,98);
	BR::BRHVec<BR::SP> b(a);
	BR::BRDVec<BR::SP> c(b);
	BR::BRHVec<BR::SP> d(c);
	BR::BRHVec<BR::SP> e(d);

	BR::BRDMat<BR::SP> f(5,98,100);
	f = 70;
	BR::BRHMat<BR::SP> g(f);
	BR::BRDMat<BR::SP> k(g);

	BR::BRHMat<BR::SP> m(k);


	printf("%f\n",e[8]);
	printf("%f\n",m[97][99]);
	return 0;
}
