#include <pycuda-complex.hpp>
#include <pycuda-helpers.hpp>
#define pi 3.14159265
#define phi 1.6180339

surface< void, cudaSurfaceType2DLayered> surf_psi ;
    
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

// //////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////  R K  -  4 //////////////////////////////////////////////////////////

// //MAP TEXTURE:      Y (up)
// //                  |      (bottom)
// //                  |    
// //                  |  
// //   (left)---------0---------X (right)
// //                  |
// //                  |
// //                  | (down)
// //          
__device__ void surfLapl(pycuda::complex<cudaPres> &result, 
                         int t_y, int t_x, 
                         cudaPres dx, cudaPres dy,
                         surface<void, cudaSurfaceType2DLayered> surf){
  pycuda::complex<cudaPres> center,  up, down, left, right ;
  fp_surf2DLayeredread( &center, surf, t_x, t_y,   int(0), cudaBoundaryModeZero ); //cudaBoundaryModeZero imply fornter condition = zero
  fp_surf2DLayeredread( &up,     surf, t_x, t_y+1, int(0), cudaBoundaryModeZero );
  fp_surf2DLayeredread( &down,   surf, t_x, t_y-1, int(0), cudaBoundaryModeZero );
  fp_surf2DLayeredread( &left,   surf, t_x-1, t_y, int(0), cudaBoundaryModeZero );
  fp_surf2DLayeredread( &right,  surf, t_x+1, t_y, int(0), cudaBoundaryModeZero );

  result  = (left + right  - 2.0cString*center )/(dx*dx) ;
  result += (up   + down   - 2.0cString*center )/(dy*dy) ;
  result *= 0.5cString;

}

__global__ void applyNablaSurface(cudaPres dx, cudaPres dy, pycuda::complex<cudaPres> *kaux){
        // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      pycuda::complex<cudaPres> kaux2;
      surfLapl(kaux2, t_i,  t_j,  dx, dy, surf_psi);
      kaux[tid] = kaux2;
}

__global__ void applyNablaSquare( int nPointX, int nPointY,  
                                  cudaPres Lx, cudaPres Ly,
                                  pycuda::complex<cudaPres> *fftpsi){
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      cudaPres k2 = 0.0;
      cudaPres kAux = KspaceFFT(t_i,nPointX, Lx);//kx[t_j];
      k2 += kAux*kAux;
      kAux = KspaceFFT(t_j,nPointY, Ly);//ky[t_i];
      k2 += kAux*kAux;
      fftpsi[tid] *= -k2;
      }
      
__global__ void energy_kernel( cudaPres xMin, cudaPres yMin, 
                           cudaPres dx, cudaPres dy,
                           cudaPres gammaX, cudaPres gammaY,
                           cudaPres dt, cudaPres constG,
                           pycuda::complex<cudaPres> *psi_state,
                           pycuda::complex<cudaPres> *V2psi ){
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      cudaPres x = t_i*dx + xMin;
      cudaPres y = t_j*dy + yMin;
      pycuda::complex<cudaPres> kiAux;
      pycuda::complex<cudaPres> psi = psi_state[tid];
      // Kinetic
      kiAux = -0.5cString * V2psi[tid] ;
      // Trap
      kiAux += 0.5cString * (x*x*gammaX*gammaX + y*y*gammaY*gammaY) *psi;
      // NL
      kiAux += 0.5cString*constG*conj(psi)*psi*psi;
      // k1
      kiAux *= conj(psi);
      V2psi[tid]=kiAux;
 }

__global__ void energy_kernelSurface( cudaPres xMin, cudaPres yMin, 
                           cudaPres dx, cudaPres dy,
                           cudaPres gammaX, cudaPres gammaY,
                           cudaPres dt, cudaPres constG,
                           pycuda::complex<cudaPres> *V2psi ){ // Assuming that surface has actual state
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      cudaPres x = t_i*dx + xMin;
      cudaPres y = t_j*dy + yMin;
      pycuda::complex<cudaPres> complexAux;
      pycuda::complex<cudaPres> psi;
      fp_surf2DLayeredread(  &psi, surf_psi,  t_j, t_i, int(0), cudaBoundaryModeZero );
      // Kinetic
      surfLapl(complexAux, t_i,  t_j,  dx, dy, surf_psi);
      complexAux = -0.5cString * complexAux ;
      // Trap
      complexAux += 0.5cString * (x*x*gammaX*gammaX + y*y*gammaY*gammaY) *psi;
      // NL
      complexAux += 0.5cString*constG*conj(psi)*psi*psi;
      // k1
      complexAux *= conj(psi);
      V2psi[tid]=complexAux;
 }
 
__global__ void rk4StepCommon( cudaPres at, cudaPres ak, cudaPres apsi, 
                           cudaPres time, cudaPres dt,
                           cudaPres xMin, cudaPres yMin, 
                           cudaPres dx, cudaPres dy, 
                           cudaPres gammaX, cudaPres gammaY,
                           cudaPres constG, 
                           pycuda::complex<cudaPres> *psi_old,
                           pycuda::complex<cudaPres> *psi_new,
                           pycuda::complex<cudaPres> *kaux, // always has k_i
                           pycuda::complex<cudaPres> *kaux2){ // always has \nabla^2 psi_old+alpha * k_{i-1}
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      cudaPres x = t_i*dx + xMin;
      cudaPres y = t_j*dy + yMin;
      //cudaPres t = time + dt*at;
      pycuda::complex<cudaPres> kiAux = kaux[tid];
      pycuda::complex<cudaPres> iComplex(0,1.0cString);
      pycuda::complex<cudaPres> psi = psi_old[tid];
      // Calculating ki
     
      // Trap +NL
      kiAux  =  0.5cString * (x*x*gammaX*gammaX + y*y*gammaY*gammaY) * kiAux + constG*conj(kiAux)*kiAux*kiAux;
      // Kinetic
      kiAux += -0.5cString * kaux2[tid] ;
      // ki
      kiAux *= -iComplex;
      
      // psi_aux for next k_i+1
      kaux[tid]     = psi + ak * dt * kiAux;
    
      // Cumulative new state
      psi_new[tid] += apsi * dt * kiAux;
      
      }
      
__global__ void rk4StepCommonSurface( cudaPres at, cudaPres ak, cudaPres apsi, 
                           cudaPres time, cudaPres dt,
                           cudaPres xMin, cudaPres yMin, 
                           cudaPres dx, cudaPres dy, 
                           cudaPres gammaX, cudaPres gammaY,
                           cudaPres constG, 
                           pycuda::complex<cudaPres> *psi_old,
                           pycuda::complex<cudaPres> *psi_new,
                           pycuda::complex<cudaPres> *kaux){ // always has \nabla^2 psi_old+alpha * k_{i-1}
      // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      cudaPres x = t_i*dx + xMin;
      cudaPres y = t_j*dy + yMin;
      //cudaPres t = time + dt*at;
      pycuda::complex<cudaPres> kiAux;
      fp_surf2DLayeredread(  &kiAux, surf_psi,  t_j, t_i, int(0), cudaBoundaryModeZero );
      pycuda::complex<cudaPres> iComplex(0,1.0cString);
      pycuda::complex<cudaPres> psi = psi_old[tid];
      // Calculating ki
     
      // Trap +NL
      kiAux  =  0.5cString * (x*x*gammaX*gammaX + y*y*gammaY*gammaY) * kiAux + constG*conj(kiAux)*kiAux*kiAux;
      // Kinetic
      kiAux += -0.5cString * kaux[tid] ;
      // ki
      kiAux *= -iComplex;
      
      // psi_aux for next k_i+1
      fp_surf2DLayeredwrite(  psi + ak * dt * kiAux, surf_psi,  t_j, t_i, int(0), cudaBoundaryModeZero );
      //kaux[tid]     = psi + ak * dt * kiAux;
    
      // Cumulative new state
      psi_new[tid] += apsi * dt * kiAux;
      
      }

__global__ void wpsi2Surf(pycuda::complex<cudaPres> *psi){
        // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      pycuda::complex<cudaPres> psistate = psi[tid];
      fp_surf2DLayeredwrite(  psistate, surf_psi, t_j, t_i, int(0), cudaBoundaryModeZero );
}
__global__ void rpsi2Surf(pycuda::complex<cudaPres> *psiaux){
        // Index for thread
      int t_i = blockIdx.x*blockDim.x + threadIdx.x;
      int t_j = blockIdx.y*blockDim.y + threadIdx.y;
      int tid = gridDim.y * blockDim.y * t_i  + t_j;
      pycuda::complex<cudaPres> psistate ;
      fp_surf2DLayeredread(  &psistate, surf_psi, t_j, t_i, int(0), cudaBoundaryModeZero );
      psiaux[tid] = psistate;
}