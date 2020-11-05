#include "SLA.h"
#include <iostream>
#include <cuda_runtime.h>
#include <string>
using namespace std;
using namespace SLA;
int main(){
    CuVec<double> v, *d_v;
    CuMat<double> m, *d_m;
    double* out, *d_out;

    out = new double[6];
	d_out = 0;
	d_v = 0;
	d_m = 0;

    cudaMalloc((void**)d_out, sizeof(double)*6);
    cudaMalloc((void**)d_m, sizeof(CuMat<double>));
    cudaMalloc((void**)d_v, sizeof(CuVec<double>));

    v.reserve(4);
    m.resize(4, 6);

    v.set(1, 2);
    m.set(0, 1, 3);
	m.set(1, 3, 0.666);

    v.upload();
    m.upload();

	v.print();
	m.print();

    cudaMemcpy(d_m, &m, sizeof(CuMat<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, &v, sizeof(CuVec<double>), cudaMemcpyHostToDevice);

    matvec<<<1, 1>>>(d_m, d_v, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, sizeof(double)*6, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < 6; ++i){
        printf("%f\n", out[i]);
    }

    delete[] out;
    cudaFree(d_out);
    m.free();
    cudaFree(d_m);
    v.free();
    cudaFree(d_v);

	string s;
	cin>>s;

    return 0;
}