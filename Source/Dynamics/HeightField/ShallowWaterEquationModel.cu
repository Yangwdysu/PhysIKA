#include "ShallowWaterEquationModel.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/MechanicalState.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Framework/Topology/FieldNeighbor.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Dynamics/ParticleSystem/Helmholtz.h"
#include "Dynamics/ParticleSystem/Attribute.h"
#include "Core/Utility.h"
#include <cuda_runtime.h>
#include "cublas_v2.h"
namespace PhysIKA
{
	IMPLEMENT_CLASS_1(ShallowWaterEquationModel, TDataType)

	template<typename TDataType>
	ShallowWaterEquationModel<TDataType>::ShallowWaterEquationModel()
		: NumericalModel()
		, m_pNum(0)
	{
		attachField(&m_position, "position", "Storing the particle positions!", false);
		attachField(&m_velocity, "velocity", "Storing the particle velocities!", false);
		//attachField(&m_force, "force", "Storing the particle force densities!", false);

		attachField(&solid, "solid", "Storing the solid grid!", false);
		attachField(&normal, "solidnormal", "Storing the solid normal!", false);
		attachField(&isBound, "isBound", "Storing the solid isBound!", false);
		attachField(&h, "h", "Storing the water height!", false);
	}

	template<typename Real, typename Coord>
	__global__ void Init(
		DeviceArray<Coord> pos,
		DeviceArray<Coord> solid,
		DeviceArray<Real> h,
		DeviceArray<int> isBound,
		DeviceArray<Coord> m_velocity
		)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= pos.size()) return;
	
		h[i] = pos[i][1] - solid[i][1];
		m_velocity[i] = Coord(0, 0, 0);
		
	}

	template<typename TDataType>
	ShallowWaterEquationModel<TDataType>::~ShallowWaterEquationModel()
	{
	}

	template<typename TDataType>
	bool ShallowWaterEquationModel<TDataType>::initializeImpl()
	{
		int num = m_position.getElementCount();
		m_accel.setElementCount(num);
		h.setElementCount(num);
		
		printf("neighbor limit is %d, index count is %d\n", neighborIndex.getReference()->getNeighborLimit(), neighborIndex.getElementCount());
		cuint pDims = cudaGridSize(num, BLOCK_SIZE);
		Init <Real, Coord> << < pDims, BLOCK_SIZE >> > (m_position.getValue(), solid.getValue(), h.getValue(), isBound.getValue(), m_velocity.getValue());
		cuSynchronize();
		return true;
	}

	template<typename Real, typename Coord>
	__global__ void computeAccel(
		DeviceArray<Real> h,
		NeighborList<int>	neighborIndex,
		DeviceArray<Coord> m_accel,
		DeviceArray<Coord> m_velocity,
		DeviceArray<Coord> m_position,
		DeviceArray<int> isBound,
		Real distance,
		Real gravity)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		int maxNei = neighborIndex.getNeighborLimit();
		
		int count1 = 0, count2 = 0;
		Real hx = 0, hz = 0;
		Real ux = 0, uz = 0, wx = 0, wz = 0;
		for (int j = 0; j < maxNei; ++j)
		{
			int nei = neighborIndex.getElement(i, j);
			if (nei >= h.size() || nei < 0)
			{
				continue;
			}
			
			if(j < maxNei/2)//gradient along z
			{
				hz += (h[nei] - h[i]) / (m_position[nei][2] - m_position[i][2]);
				uz += (m_velocity[nei][0] - m_velocity[i][0]) / (m_position[nei][2] - m_position[i][2]);
				wz += (m_velocity[nei][2] - m_velocity[i][2]) / (m_position[nei][2] - m_position[i][2]);
				count1++;
			}
			else
			{
				hx += (h[nei] - h[i]) / (m_position[nei][0] - m_position[i][0]);
				ux += (m_velocity[nei][0] - m_velocity[i][0]) / (m_position[nei][0] - m_position[i][0]);
				wx += (m_velocity[nei][2] - m_velocity[i][2]) / (m_position[nei][0] - m_position[i][0]);
				count2++;
			}
		}
		
		//m_accel[i][0] = -(gravity * hx + m_velocity[i][0] * ux) / count2 - m_velocity[i][2] * uz / count1;
		//m_accel[i][2] = -(gravity * hz + m_velocity[i][2] * wz) / count1 - m_velocity[i][0] * wx / count2;
		//m_accel[i][0] = -gravity * hx / count2;
		//m_accel[i][2] = -gravity * hz / count1;

		m_accel[i][0] = -(gravity * hx + m_velocity[i][0] * ux) / 2 - m_velocity[i][2] * uz / 2;
		m_accel[i][2] = -(gravity * hz + m_velocity[i][2] * wz) / 2 - m_velocity[i][0] * wx / 2;
		

	}

	template<typename Real, typename Coord>
	__global__ void computeVelocity(
		DeviceArray<Real> h,
		NeighborList<int>	neighborIndex,
		DeviceArray<Coord> m_accel,
		DeviceArray<Coord> m_velocity,
		DeviceArray<Coord> m_position,
		DeviceArray<int> isBound,
		Real distance,
		Real relax,
		Real gravity,
		Real dt)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		//if (isBound[i])return;
		int maxNei = neighborIndex.getNeighborLimit();
		
		int count1 = 0, count2 = 0;
		Real hx = 0, hz = 0;
		for (int j = 0; j < maxNei; ++j)
		{
			int nei = neighborIndex.getElement(i, j);
			//bound cell
			if (nei >= h.size() || nei < 0)
			{
				continue;
			}
			if (j < maxNei / 2)//gradient along z
			{
				hz += (m_accel[nei][2] - m_accel[i][2]) / (m_position[nei][2] - m_position[i][2]);
				count1++;
			}
			else
			{
				hx += (m_accel[nei][0] - m_accel[i][0]) / (m_position[nei][0] - m_position[i][0]);
				count2++;
			}
		}
		
		Real maxVel = 2*sqrt(distance*gravity), vel;
		//m_velocity[i][1] = ((hz / count1 + hx / count2)*gravity*distance)*dt+ m_velocity[i][1];
		m_velocity[i][1] = ((hz / 2 + hx / 2)*gravity*distance)*dt + m_velocity[i][1];
		m_velocity[i][0] = relax * m_velocity[i][0] + dt * m_accel[i][0];
		m_velocity[i][2] = relax * m_velocity[i][2] + dt * m_accel[i][2];

		vel = sqrt(pow(m_velocity[i][0], 2) + pow(m_velocity[i][2], 2));
		if (vel > maxVel)
		{
			m_velocity[i][0] *= maxVel / vel;
			m_velocity[i][2] *= maxVel / vel;
		}
	}

	template<typename Real, typename Coord>
	__global__ void computeBoundConstrant(
		DeviceArray<Real> h,
		NeighborList<int>	neighborIndex,
		DeviceArray<Coord> m_accel,
		DeviceArray<Coord> m_velocity,
		DeviceArray<Coord> m_position,
		DeviceArray<int> isBound,
		Real distance,
		Real gravity,
		Real dt)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		if (isBound[i] == 0)return;
		int maxNei = neighborIndex.getNeighborLimit();
		int count1 = 0, count2 = 0;
		Real hx = 0, hz = 0;
		int znei, xnei;
		for (int j = 0; j < maxNei; ++j)
		{
			int nei = neighborIndex.getElement(i, j);
			if (nei >= h.size() || nei < 0)
			{
				switch(j)
				{
				case 0:
					//if (m_velocity[i][2] < 0)
						m_velocity[i][2] *= 0;
						//m_velocity[i][2] = -0.1*m_velocity[i][2];
					break;
				case 1:
					//if (m_velocity[i][2] > 0)
						m_velocity[i][2] *= 0;
						//m_velocity[i][2] = -0.1*m_velocity[i][2];
					break;
				case 2:
					//if (m_velocity[i][0] < 0)
						m_velocity[i][0] *= 0;
						//m_velocity[i][0] = -0.1*m_velocity[i][0];
					break;
				case 3:
					//if (m_velocity[i][0] > 0)
						m_velocity[i][0] *= 0;
						//m_velocity[i][0] = -0.1*m_velocity[i][0];
					break;
				}
			}
			
		}
	}

	template<typename Real, typename Coord>
	__global__ void computeHeight(
		DeviceArray<Real> h,
		NeighborList<int>	neighborIndex,
		DeviceArray<Coord> m_velocity,
		DeviceArray<Coord> m_accel,
		DeviceArray<int> isBound,
		DeviceArray<Coord> m_position,
		DeviceArray<Coord> solid,
		DeviceArray<Coord> normal,
		Real distance,
		Real dt)
	{
		int i = threadIdx.x + (blockIdx.x * blockDim.x);
		if (i >= h.size())  return;
		int maxNei = neighborIndex.getNeighborLimit();

		int count1 = 0, count2 = 0;
		Real uhx = 0, whz = 0;
		for (int j = 0; j < maxNei; ++j)
		{
			int nei = neighborIndex.getElement(i, j);
			//bound cell
			if (nei >= h.size() || nei < 0)
			{
				continue;
			}
			if (j < maxNei / 2)//gradient along z
			{
				whz += (h[nei]*m_velocity[nei][2]-h[i]*m_velocity[i][2]) / (m_position[nei][2] - m_position[i][2]);
				count1++;
			}
			else
			{
				uhx += (h[nei] * m_velocity[nei][0] - h[i] * m_velocity[i][0]) / (m_position[nei][0] - m_position[i][0]);
				count2++;
			}
		}
		
		h[i] += -(uhx / 2 + whz / 2)*dt;
		m_position[i][1] = solid[i][1] + h[i];

		if (i == 0)
		{
			printf("Height: %f \n", m_position[i][1]);
		}
	}

	template<typename TDataType>
	void ShallowWaterEquationModel<TDataType>::step(Real dt)
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Parent not set for ParticleSystem!");
			return;
		}
		
		int num = m_position.getElementCount();
		cuint pDims = cudaGridSize(num, BLOCK_SIZE);

		computeBoundConstrant <Real, Coord> << < pDims, BLOCK_SIZE >> > (
			h.getValue(),
			neighborIndex.getValue(),
			m_accel.getValue(),
			m_velocity.getValue(),
			m_position.getValue(),
			isBound.getValue(),
			distance,
			9.8,
			dt
			);
		cuSynchronize();

		computeAccel <Real, Coord> << < pDims, BLOCK_SIZE >> > (
			h.getValue(),
			neighborIndex.getValue(),
			m_accel.getValue(),
			m_velocity.getValue(),
			m_position.getValue(),
			isBound.getValue(),
			distance,
			9.8
			);
		cuSynchronize();

		computeVelocity <Real, Coord> << < pDims, BLOCK_SIZE >> > (
			h.getValue(),
			neighborIndex.getValue(),
			m_accel.getValue(),
			m_velocity.getValue(),
			m_position.getValue(),
			isBound.getValue(),
			distance,
			relax,
			9.8,
			dt
			);
		cuSynchronize();



		computeHeight <Real, Coord> << < pDims, BLOCK_SIZE >> > (
			h.getValue(),
			neighborIndex.getValue(),
			m_velocity.getValue(),
			m_accel.getValue(),
			isBound.getValue(),
			m_position.getValue(),
			solid.getValue(),
			normal.getValue(),
			distance,
			dt
			);
		cuSynchronize();

	/*	cublasHandle_t handle;
		float sum;
		cublasCreate(&handle);
		cublasSasum(handle, solid.getElementCount(), h.getValue().getDataPtr(), 1, &sum);
		cublasDestroy(handle);
		printf("total height is %f\n", sum);*/
	}
}
