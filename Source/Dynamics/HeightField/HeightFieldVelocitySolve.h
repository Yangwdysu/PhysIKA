
#include "Framework/Framework/NumericalIntegrator.h"
//#include"Platform.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Array/Array4D.h"
#include"ParticleSystem/ParticleSystem.h"
#include "Dynamics/RigidBody/RigidBody.h"


namespace PhysIKA {
#define BLOCKSIZE_X 16
#define BLOCKSIZE_Y 16
	template<typename TDataType>
	class HeightFieldVelocitySolve:public Node {
		DECLARE_CLASS_1(HeightFieldVelocitySolve, TDataType)

	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;
		typedef typename TDataType::Matrix Matrix;

		HeightFieldVelocitySolve();
		~HeightFieldVelocitySolve();
		void animate(float dt);
		void swapDeviceGrid();
		void initialized(int size, float patchLength);
		void initDynamicRegion();
		void initSource();
		void addSource();



	public:
		DeviceArrayField<Coord> m_Height;
		DeviceArrayField<Coord> m_Rvelocity;
		DeviceArrayField<Coord> m_Lvelocity;

		Grid4f m_device_grid;
		Grid4f m_device_grid_next;
		Grid4f m_height;
		Grid2f m_source;
		Real m_horizon;
		//float m_horizon = 2.0f;


		float m_patch_length;
		float m_realGridSize;			//网格实际距离
		int pathcSize;

		int m_simulatedRegionWidth;		//动态区域宽度
		int m_simulatedRegionHeight;	//动态区域高度


		int m_simulatedOriginX = 0;			//动态区域初始x坐标
		int m_simulatedOriginY = 0;			//动态区域初始y坐标

	};



#ifdef PRECISION_FLOAT
	template class HeightFieldVelocitySolve<DataType3f>;
#else
	template class HeightFieldVelocitySolve<DataType3d>;
#endif

}