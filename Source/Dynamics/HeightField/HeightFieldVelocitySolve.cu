#include <cuda_runtime.h>

#include"HeightFieldVelocitySolve.h"


namespace PhysIKA {



	__constant__ float GRAVITY = 9.83219f * 0.5f; //0.5f * Fallbeschleunigung

	template<typename TDataType>
	inline HeightFieldVelocitySolve<TDataType>::HeightFieldVelocitySolve()
	{
	}

	template<typename TDataType>
	void HeightFieldVelocitySolve<TDataType>::initialized(int size, float patchLength)
	{
		m_patch_length = patchLength;//512
		m_realGridSize = patchLength / size;//512/512=1

		m_simulatedRegionWidth = size;//512
		m_simulatedRegionHeight = size;

		m_simulatedOriginX = 0;
		m_simulatedOriginY = 0;

		initialized();
		initDynamicRegion();

		initSource();
	}


	__global__ void C_InitDynamicRegion(float4 *grid, int gridwidth, int gridheight, int pitch, float level)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < gridwidth && y < gridheight)
		{
			float4 gp;
			gp.x = level;//level=2.0
			gp.y = 0.0f;
			gp.z = 0.0f;
			gp.w = 0.0f;

			grid[x + y*gridwidth] = gp;
			//grid2Dwrite(grid, x, y, pitch, gp);
		}
	}
	template<typename TDataType>
	void  HeightFieldVelocitySolve<TDataType>::initDynamicRegion()
	{
		cudaError_t error;

		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

		size_t pitch;

		int num = extNx*extNy;

		m_device_grid.Resize(extNx, extNy,0);
		m_device_grid_next.Resize(extNx, extNy, 0);
		m_height.Resize(m_simulatedRegionWidth,m_simulatedRegionHeight,0);
		//cudaCheck(cudaMallocPitch(&m_device_grid, &pitch, extNx * sizeof(gridpoint), extNy));
		//cudaCheck(cudaMallocPitch(&m_device_grid_next, &pitch, extNx * sizeof(gridpoint), extNy));
		//cudaCheck(cudaMalloc((void **)&m_height, m_simulatedRegionWidth*m_simulatedRegionWidth * sizeof(float4)));

		//gl_utility::createTexture(m_simulatedRegionWidth, m_simulatedRegionHeight, GL_RGBA32F, m_height_texture, GL_CLAMP_TO_BORDER, GL_LINEAR, GL_LINEAR, GL_RGBA, GL_FLOAT);
		//cudaCheck(cudaGraphicsGLRegisterImage(&m_cuda_texture, m_height_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

		//m_grid_pitch = pitch / sizeof(gridpoint);

		int x = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		//init grid with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (m_device_grid, extNx, extNy, m_grid_pitch, m_horizon);
		//synchronCheck;

		//init grid_next with initial values
		C_InitDynamicRegion << < blocksPerGrid, threadsPerBlock >> > (m_device_grid_next, extNx, extNy, m_grid_pitch, m_horizon);
		//synchronCheck;

		error = cudaThreadSynchronize();

		//g_cpChannelDesc = cudaCreateChannelDesc<float4>();
		//cudaCheck(cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint)));
	}


	__global__ void C_InitSource(
		float2* source,
		int patchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			if (i < patchSize / 2 + 3 && i > patchSize / 2 - 3 && j < patchSize / 2 + 3 && j > patchSize / 2 - 3)
			{
				float2 uv = make_float2(1.0f);
				source[i + j*patchSize] = uv;
			}
		}
	}

	template<typename TDataType>
	void HeightFieldVelocitySolve<TDataType>::initSource()
	{
		int sizeInBytes = m_simulatedRegionWidth* m_simulatedRegionHeight * sizeof(float2);

		//cudaCheck(cudaMalloc(&m_source, sizeInBytes));
		//cudaCheck(cudaMalloc(&m_weight, m_simulatedRegionWidth* m_simulatedRegionHeight * sizeof(float)));
		//cudaCheck(cudaMemset(m_source, 0, sizeInBytes));

		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);
		C_InitSource << < blocksPerGrid, threadsPerBlock >> > (m_source, m_simulatedRegionWidth);
		//resetSource();
		//synchronCheck;
	}

	__device__ float C_GetU(float4 gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;

		float h4 = h * h * h * h;
		return sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));
	}
	__device__ float C_GetV(float4 gp)
	{
		float h = max(gp.x, 0.0f);
		float vh = gp.z;

		float h4 = h * h * h * h;
		return sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));
	}

	__global__ void C_AddSource(
		float4 *grid,
		float2* source,
		int patchSize,
		int pitchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int gx = i + 1;
			int gy = j + 1;

			float4 gp = grid[gx+gy*pitchSize];
			float2 s_ij = source[i + j*patchSize];

			float h = gp.x;
			float u = C_GetU(gp);
			float v = C_GetV(gp);

			if (length(s_ij) > 0.001f)
			{
				u += s_ij.x;
				v += s_ij.y;

				u *= 0.98f;
				v *= 0.98f;

				u = min(0.4f, max(-0.4f, u));
				v = min(0.4f, max(-0.4f, v));
			}

			gp.x = h;
			gp.y = u*h;
			gp.z = v*h;
			grid[gx + gy*pitchSize]= gp ;
			//grid2Dwrite(grid, gx, gy, pitchSize, gp);
		}
	}
	template<typename TDataType>
	void HeightFieldVelocitySolve<TDataType>::addSource()
	{
		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		//cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, m_simulatedRegionWidth + 2, m_simulatedRegionHeight + 2, m_grid_pitch * sizeof(gridpoint));
		C_AddSource << < blocksPerGrid, threadsPerBlock >> > (
			m_device_grid_next,
			m_source,
			m_simulatedRegionWidth,
			m_grid_pitch);
		swapDeviceGrid();
		//synchronCheck;
	}
	__global__ void C_ImposeBC(float4* grid_next, float4* grid, int width, int height, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			if (x == 0)
			{
				float4 a = grid[1 + y*width];
				grid_next[x + y*width] = a;
			}
			else if (x == width - 1)
			{
				float4 a = grid[(width - 2) + y*width];
				grid_next[x + y*width] = a;
			}
			else if (y == 0)
			{
				float4 a = grid[x + 1 * width];
				grid_next[x + y*width] = a;
			}
			else if (y == height - 1)
			{
				float4 a = grid[x + (height - 2)*width];
				grid_next[x + y*width] = a;
			}
			else
			{
				float4 a = grid[x + y*width];
				grid_next[x + y*width] = a;
			}
		}
	}


	__host__ __device__ void C_FixShore(float4 &l, float4 &c, float4 &r)
	{
		l.x = 0;
		if (r.x < 0.0f || l.x < 0.0f || c.x < 0.0f)
		{
			
			c.x = c.x + l.x + r.x;
			c.x = max(0.0f, c.x);
			l.x = 0.0f;
			r.x = 0.0f;
		}
		Real h = c.x;
		Real h4 = h * h * h * h;
		Real v = sqrtf(2.0f) * h * c.y / (sqrtf(h4 + max(h4, EPSILON)));
		Real u = sqrtf(2.0f) * h * c.z / (sqrtf(h4 + max(h4, EPSILON)));
		c.y = u * h;
		c.z = v * h;
	}


	__host__ __device__ float4 C_VerticalPotential(float4 gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float v = sqrtf(2.0f) * h * vh / (sqrtf(h4 + max(h4, EPSILON)));
		float4 G;
		G.x = v * h;
		G.y = uh * v;
		G.z = vh * v + GRAVITY * h * h;
		G.w = 0.0f;
		return G;
	}

	__device__ float4 C_HorizontalPotential(float4 gp)
	{
		float h = max(gp.x, 0.0f);
		float uh = gp.y;
		float vh = gp.z;

		float h4 = h * h * h * h;
		float u = sqrtf(2.0f) * h * uh / (sqrtf(h4 + max(h4, EPSILON)));

		float4 F;
		F.x = u * h;
		F.y = uh * u + GRAVITY * h * h;
		F.z = vh * u;
		F.w = 0.0f;
		return F;
	}
	__device__ float4 C_SlopeForce(float4 c, float4 n, float4 e, float4 s, float4 w)
	{
		float h = max(c.x, 0.0f);

		float4 H;
		H.x = 0.0f;
		H.y = -GRAVITY * h * (e.w - w.w);
		H.z = -GRAVITY * h * (s.w - n.w);
		H.w = 0.0f;
		return H;
	}
	__global__ void C_OneWaveStep(float4* grid, float4* grid_next,int width, int height, Real timestep, int pitch)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		if (x < width && y < height)
		{
			int gridx = x + 1;
			int gridy = y + 1;

			float4 center = grid[gridx+gridy*width];

			float4 north = grid[gridx + (gridy - 1)*width];

			float4 west = grid[(gridx - 1) + gridy*width];

			float4 south = grid[gridx + (gridy + 1)*width];

			float4 east = grid[(gridx + 1) + gridy*width];

			C_FixShore(west, center, east);
			C_FixShore(north, center, south);

			float4 u_south = 0.5f * (south + center) - timestep * (C_VerticalPotential(south) - C_VerticalPotential(center));
			float4 u_north = 0.5f * (north + center) - timestep * (C_VerticalPotential(center) - C_VerticalPotential(north));
			float4 u_west = 0.5f * (west + center) - timestep * (C_HorizontalPotential(center) - C_HorizontalPotential(west));
			float4 u_east = 0.5f * (east + center) - timestep * (C_HorizontalPotential(east) - C_HorizontalPotential(center));

			float4 u_center = center + timestep * C_SlopeForce(center, north, east, south, west) - timestep * (C_HorizontalPotential(u_east) - C_HorizontalPotential(u_west)) - timestep * (C_VerticalPotential(u_south) - C_VerticalPotential(u_north));
			u_center.x = max(0.0f, u_center.x);

			grid_next[gridx + gridy*width] = u_center;
		}
	}


	__global__ void C_InitHeightField(
		float4* height,
		float4* grid,
		int width,
		int patchSize,
		float horizon,
		float realSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int gridx = i + 1;
			int gridy = j + 1;

			float4 gp = grid[gridx+gridy*width];
			height[i + j*patchSize].x = gp.x - horizon;

			float d = sqrtf((i - patchSize / 2)*(i - patchSize / 2) + (j - patchSize / 2)*(j - patchSize / 2));
			float q = d / (0.49f*patchSize);

			float weight = q < 1.0f ? 1.0f - q * q : 0.0f;
			height[i + j*patchSize].y = 1.3f * realSize * sinf(3.0f*weight*height[i + j * patchSize].x*0.5f*M_PI);

			// x component stores the original height, y component stores the normalized height, z component stores the X gradient, w component stores the Z gradient;
		}
	}

	__global__ void C_InitHeightGrad(
		float4* height,
		int patchSize)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		int j = threadIdx.y + blockIdx.y * blockDim.y;
		if (i < patchSize && j < patchSize)
		{
			int i_minus_one = (i - 1 + patchSize) % patchSize;
			int i_plus_one = (i + 1) % patchSize;
			int j_minus_one = (j - 1 + patchSize) % patchSize;
			int j_plus_one = (j + 1) % patchSize;

			float4 Dx = (height[i_plus_one + j * patchSize] - height[i_minus_one + j * patchSize]) / 2;
			float4 Dz = (height[i + j_plus_one * patchSize] - height[i + j_minus_one * patchSize]) / 2;

			height[i + patchSize * j].z = Dx.y;
			height[i + patchSize * j].w = Dz.y;
		}
	}
	template<typename TDataType>
	void HeightFieldVelocitySolve<TDataType>::animate(float dt)
	{
		
		int extNx = m_simulatedRegionWidth + 2;
		int extNy = m_simulatedRegionHeight + 2;

		cudaError_t error;
		// make dimension
		int x = (m_simulatedRegionWidth + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y = (m_simulatedRegionHeight + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid(x, y);

		int x1 = (extNx + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
		int y1 = (extNy + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
		dim3 threadsPerBlock1(BLOCKSIZE_X, BLOCKSIZE_Y);
		dim3 blocksPerGrid1(x1, y1);

		int nStep = 1;
		float timestep = dt / nStep;


		for (int iter = 0; iter < nStep; iter++)
		{
			//cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));
			C_ImposeBC << < blocksPerGrid1, threadsPerBlock1 >> > (m_device_grid_next, m_device_grid, extNx, extNy, m_grid_pitch);
			swapDeviceGrid();
			//synchronCheck;

			//cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));

			C_OneWaveStep << < blocksPerGrid, threadsPerBlock >> > (
				m_device_grid_next,
				m_simulatedRegionWidth,
				m_simulatedRegionHeight,
				1.0f*timestep,
				m_grid_pitch);
			swapDeviceGrid();
			//synchronCheck;
		}



		//error = cudaBindTexture2D(0, &g_capillaryTexture, m_device_grid, &g_cpChannelDesc, extNx, extNy, m_grid_pitch * sizeof(gridpoint));
		C_InitHeightField << < blocksPerGrid, threadsPerBlock >> > (m_height, m_simulatedRegionWidth, m_horizon, m_realGridSize);
		//synchronCheck;
		C_InitHeightGrad << < blocksPerGrid, threadsPerBlock >> > (m_height, m_simulatedRegionWidth);
		//synchronCheck;

		//cudaCheck(cudaGraphicsMapResources(1, &m_cuda_texture));
		cudaArray* cuda_height_array = nullptr;
		//cudaCheck(cudaGraphicsSubResourceGetMappedArray(&cuda_height_array, m_cuda_texture, 0, 0));
		//cudaCheck(cudaMemcpyToArray(cuda_height_array, 0, 0, m_height, m_simulatedRegionWidth*m_simulatedRegionHeight * sizeof(float4), cudaMemcpyDeviceToDevice));
		//cudaCheck(cudaGraphicsUnmapResources(1, &m_cuda_texture));
	}

	template<typename TDataType>
	void HeightFieldVelocitySolve<TDataType>::swapDeviceGrid()
	{
		Grid4f grid_helper = m_device_grid;
		m_device_grid = m_device_grid_next;
		m_device_grid_next = grid_helper;
	}


}