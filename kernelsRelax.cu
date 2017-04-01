#include <pycuda-complex.hpp>
#define pi 3.14159265
#define phi 1.6180339

__device__ cudaPres KspaceFFT(int tid, int nPoint, cudaPres L){
cudaPres Kfft;
if (tid < nPoint/2){
  Kfft = 2.0f*pi*(tid)/L;
  }
else {
Kfft = 2.0f*pi*(tid-nPoint)/L;
}
return Kfft;
}

__global__ void curandinitComplex_kernel( cudaPres dx, cudaPres dy,
      cudaPres xMin, cudaPres yMin, 
      cudaPres gammaX, cudaPres gammaY, 
      cudaPres *rndReal, cudaPres *rndImag,  pycuda::complex<cudaPres> *psi){
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      // Indices to coordinates
      cudaPres x = t_i*dx + xMin;
      cudaPres y = t_j*dy + yMin;
      pycuda::complex<cudaPres> auxC;
      cudaPres aux = exp(-gammaX*gammaX*x*x-gammaY*gammaY*y*y);
      auxC._M_re = aux + aux*(rndReal[tid]-0.5cString);
      auxC._M_im = aux + aux*(rndImag[tid]-0.5cString);
      psi[tid] = auxC;
}


__global__ void getAlphas_kernel( cudaPres dx, cudaPres dy, 
cudaPres xMin, cudaPres yMin, 
cudaPres gammaX, cudaPres gammaY,   cudaPres constG,
pycuda::complex<cudaPres> *psi1, pycuda::complex<cudaPres> *alphas){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid = gridDim.y * blockDim.y * t_i  + t_j;
cudaPres result = 0.0;
cudaPres ri = t_i*dx + xMin;
result += 0.5*gammaX*gammaX*ri*ri;

ri = t_j*dy + yMin;
result += 0.5*gammaY*gammaY*ri*ri;

ri = abs(psi1[tid]);
alphas[tid] =  result + constG*ri*ri;

}

__global__ void implicitStep1_kernel( cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy,  
cudaPres alpha,  cudaPres gammaX, cudaPres gammaY, cudaPres constG,
 pycuda::complex<cudaPres> *psi1_d){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid = gridDim.y * blockDim.y * t_i  + t_j;

cudaPres result = 0.0;
cudaPres ri = t_i*dx + xMin;
result += 0.5*gammaX*gammaX*ri*ri;

ri = t_j*dy + yMin;
result += 0.5*gammaY*gammaY*ri*ri;

pycuda::complex<cudaPres> psi1; //,Vtrap,torque;//psi1, psi2, psi3, partialX, partialY, Vtrap, torque, lz, result;

psi1 = psi1_d[tid];
ri = abs(psi1);
result *= -1;
result += alpha;
result -= constG*ri*ri;

psi1_d[tid] = psi1*result;
 }

__global__ void implicitStep2_kernel( cudaPres dt, cudaPres alpha, 
int nPointX, int nPointY,  cudaPres Lx, cudaPres Ly, 
pycuda::complex<cudaPres> *psiTransf, pycuda::complex<cudaPres> *GTranf){
int t_i = blockIdx.x*blockDim.x + threadIdx.x;
int t_j = blockIdx.y*blockDim.y + threadIdx.y;
int tid = gridDim.y * blockDim.y * t_i  + t_j;

cudaPres k2 = 0.0;
cudaPres kAux = KspaceFFT(t_i,nPointX, Lx);//kx[t_j];
k2 += kAux*kAux;
kAux = KspaceFFT(t_j,nPointY, Ly);//ky[t_i];
k2 += kAux*kAux;

pycuda::complex<cudaPres> factor, psiT, Gt;
// factor = 2.0 / ( 2.0 + dt*(k2 + 2.0*alpha) );
kAux = 2.0 / ( 2.0 + dt*(k2 + 2.0*alpha) );
psiT = psiTransf[tid];
Gt = GTranf[tid];
psiTransf[tid] = kAux * ( psiT + Gt*dt);

}
    
 ///////////////////////////////////////////////////////////
    
      __global__ void strangL_kernel( int nx, int ny,
      cudaPres dtau,
      cudaPres Lx, cudaPres Ly,
      pycuda::complex<cudaPres> *fftpsi){
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      // Indices to coordinates
      cudaPres kx = KspaceFFT( t_i,  nx,  Lx);
      cudaPres ky = KspaceFFT( t_j,  ny,  Ly);

      fftpsi[tid] *= exp(-dtau*0.25cString*(kx*kx+ky*ky));
      
      }
      
      __global__ void strangNL_kernel(cudaPres dtau, cudaPres gNonLinear,
      cudaPres dx, cudaPres dy,
      cudaPres xMin, cudaPres yMin,
      cudaPres gammaX, cudaPres gammaY,
      pycuda::complex<cudaPres> *psi){
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      // Indices to coordinates
      cudaPres x = t_i*dx + xMin;
      cudaPres y = t_j*dy + yMin;
      pycuda::complex<cudaPres> auxC=psi[tid];
      auxC = (x*x*gammaX*gammaX+y*y*gammaY*gammaY)*0.5cString+gNonLinear*conj(auxC)*auxC;
      psi[tid] *= exp(-dtau*auxC._M_re);  
      }
