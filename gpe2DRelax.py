from numpy import float32, complex64,float64, complex128, array, int32, sqrt, pi
from pycuda import gpuarray
from pycuda import autoinit
from pycuda.compiler import SourceModule #Functions in CUDA
from pycuda import curandom
from pyfft.cuda import Plan ## FFT in CUDA
from pycuda.elementwise import ElementwiseKernel
from matplotlib.pylab import subplots, colorbar

# Precision
prec = float32 #In CUDA is imprtant the type
precComplex = complex64
cuPrec = 'float'
cString = 'f'

# Box
Nx = Ny = int32(256)
Lx = Ly = prec(12.8*2) #At hand!
x_min , y_min = prec(-Lx/2.0) , prec(-Ly/2.0)
dx = prec( Lx/(Nx-1))
dy = prec(Ly/(Ny-1))
#dx,dy
# System 
g=prec(70 * sqrt(200/(pi)))#Colmillo!
a = prec(200)
dtau = prec(0.001)
gammaX = prec(1.0)
gammaY = prec(0.85)

# Distributing work in threads
blockDims=(32,32,1) # Threads in Block FREE
if blockDims[0]*blockDims[1] < 1024: # GPU architecture
    print 'Max threads per block are 1024'
    raise Exception('Max threads per block are 1024')
    
gridDims = (Nx//blockDims[0]+1*(Nx%blockDims[0]!=0),
           Ny//blockDims[1]+1*(Ny%blockDims[1]!=0),1)

# Useful functions
gpuFFT = Plan((Nx,Ny),dtype=precComplex)
def compileGPU(stringKernel): #Specific function
    stringKernel = stringKernel.replace('cudaPres', cuPrec)
    stringKernel = stringKernel.replace('cString', cString)
    return SourceModule(stringKernel)

# PyCUDA nice functions!
mult_C = ElementwiseKernel(arguments="{0} a, pycuda::complex<{0}> *psi".format( cuPrec ),
				operation = "psi[i] = a*psi[i] ",
				name = "multiplyByFloat_kernel",
				preamble="#include <pycuda-complex.hpp>")
copy_C = ElementwiseKernel(arguments="pycuda::complex<{0}> *psi1, pycuda::complex<{0}> *psi2".format( cuPrec ),
				operation = "psi2[i] = psi1[i] ",
				name = "multiplyByFloat_kernel",
				preamble="#include <pycuda-complex.hpp>")
from pycuda.reduction import ReductionKernel
get_Norm_C = ReductionKernel( prec,
				neutral = "0",
				arguments=" {0} dx, {0} dy,  pycuda::complex<{0}> * psi ".format(cuPrec),
				map_expr = "( conj(psi[i])* psi[i] )._M_re*dx*dy",
				reduce_expr = "a+b",
				name = "getNorm_kernel",
				preamble="#include <pycuda-complex.hpp>")
get_integral_C = ReductionKernel( precComplex,
				neutral = "0",
				arguments=" {0} dx, {0} dy,  pycuda::complex<{0}> * psi ".format(cuPrec),
				map_expr = "psi[i] * dx * dy",
				reduce_expr = "a+b",
				name = "getNorm_kernel",
				preamble="#include <pycuda-complex.hpp>")

def plotState(psi):
    f,ax = subplots(1,1, figsize=(12,12))
    dens = ax.imshow(psi,extent=[x_min,x_min+dx*Nx,y_min,y_min+dy*Ny], cmap='magma', origin='lower')
    colorbar(dens)
# My function to normalize
def normalizeGPU(psiGPU):
    norm = get_Norm_C(dx,dy,psiGPU).get()
    norm = 1/sqrt(norm)
    mult_C(norm,psiGPU)
    return norm 
    
fileKernels = open('kernelsRelax.cu')
kernels = fileKernels.read()
fileKernels.close()
compiledKernels = compileGPU(kernels)

step1 =compiledKernels.get_function('implicitStep1_kernel')
step2 =compiledKernels.get_function('implicitStep2_kernel')
getAlphas = compiledKernels.get_function('getAlphas_kernel')
initGPU = compiledKernels.get_function( "curandinitComplex_kernel" )

initGPU.prepare(cString*6+'PPP')
step1.prepare('ffffffffP')#cudaPres xMin, cudaPres yMin, cudaPres dx, cudaPres dy,  
#cudaPres alpha,  cudaPres gammaX, cudaPres gammaY, cudaPres constG,
# pycuda::complex<cudaPres> *psi1_d
step2.prepare('ffiiffPP')#cudaPres dt, cudaPres alpha, 
#int nPointX, int nPointY,  cudaPres Lx, cudaPres Ly, 
#pycuda::complex<cudaPres> *psiTransf, pycuda::complex<cudaPres> *GTranf
getAlphas.prepare('fffffffPP') #cudaPres dx, cudaPres dy, 
#cudaPres xMin, cudaPres yMin, 
#cudaPres gammaX, cudaPres gammaY,  cudaPres constG,
#pycuda::complex<cudaPres> *psi1, pycuda::complex<cudaPres> *alphas

#Initializing State and MEMORY Allocations
Psi_gpu     = gpuarray.zeros([Nx,Ny], dtype=precComplex)
Psi_gpu_k   = gpuarray.zeros([Nx,Ny], dtype=precComplex)

cudarandom = curandom.XORWOWRandomNumberGenerator() #x60! , see pycuda Docs!
def initState():
        real_gpu = cudarandom.gen_uniform([Nx,Ny],prec) # Random GPU array
        imag_gpu = cudarandom.gen_uniform([Nx,Ny],prec)
        # Calling function, executing
        initGPU.prepared_call(gridDims, blockDims,
                      dx,dy,
                     x_min,y_min,
                     gammaX,gammaY,
                     real_gpu.gpudata,imag_gpu.gpudata,Psi_gpu.gpudata)

        real_gpu.gpudata.free() # Delete arrays on GPU, freeing memory
        imag_gpu.gpudata.free()

initState()
        
def implicitRelaxStep():
    global a
    gpuFFT.execute(Psi_gpu,Psi_gpu_k)
    step1.prepared_call(gridDims, blockDims,x_min,y_min, 
                    dx,dy,a,gammaX,gammaY,
                    g,Psi_gpu.gpudata)
    gpuFFT.execute(Psi_gpu)
    step2.prepared_call(gridDims, blockDims, dtau, a, Nx, Ny, Lx,Ly,
                    Psi_gpu_k.gpudata, Psi_gpu.gpudata)
    gpuFFT.execute(Psi_gpu_k,Psi_gpu,inverse=True)
    normalizeGPU(Psi_gpu)
    getAlphas.prepared_call(gridDims, blockDims,dx,dy,
                        x_min,y_min,
                        gammaX,gammaY,g,
                        Psi_gpu.gpudata,Psi_gpu_k.gpudata)
    a = 0.5*(gpuarray.max(Psi_gpu_k.real).get()+gpuarray.max(Psi_gpu_k.imag).get())
    


strangLinear  = compiledKernels.get_function('strangL_kernel')
strangNLinear = compiledKernels.get_function('strangNL_kernel')
strangLinear.prepare('iifffP')
strangNLinear.prepare('ffffffffP')

def splitStrangRelaxStep():
    gpuFFT.execute(Psi_gpu)
    strangLinear.prepared_call(gridDims, blockDims,
                           int32(Nx),int32(Ny),
                           prec(dtau),prec(Lx),prec(Ly),
                           Psi_gpu.gpudata)
    gpuFFT.execute(Psi_gpu, inverse=True)
    strangNLinear.prepared_call(gridDims, blockDims,
                           prec(dtau),prec(g),
                           prec(dx),prec(dy),
                           prec(x_min),prec(y_min),
                           prec(gammaX),prec(gammaY),
                           Psi_gpu.gpudata)
    gpuFFT.execute(Psi_gpu)
    strangLinear.prepared_call(gridDims, blockDims,
                           int32(Nx),int32(Ny),
                           prec(dtau),prec(Lx),prec(Ly),
                           Psi_gpu.gpudata)
    gpuFFT.execute(Psi_gpu, inverse=True)
    norma = get_Norm_C(prec(dx),prec(dy),Psi_gpu).get()
    #print norma 
    inv = 1/sqrt(norma)
    mult_C(inv,Psi_gpu)