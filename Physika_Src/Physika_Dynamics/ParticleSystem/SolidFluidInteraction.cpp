#include "SolidFluidInteraction.h"
#include "PositionBasedFluidModel.h"

#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Render/PointRenderModule.h"
#include "Physika_Core/Utility.h"


namespace Physika
{
	IMPLEMENT_CLASS_1(SolidFluidInteraction, TDataType)

	template<typename TDataType>
	SolidFluidInteraction<TDataType>::SolidFluidInteraction(std::string name)
		: Node(name)
	{
		setName("default");
	}

	template<typename TDataType>
	SolidFluidInteraction<TDataType>::~SolidFluidInteraction()
	{
		
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::initialize()
	{
		
		return true;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addRigidBody(std::shared_ptr<RigidBody<TDataType>> child)
	{
		return false;
	}

	template<typename TDataType>
	bool SolidFluidInteraction<TDataType>::addParticleSystem(std::shared_ptr<ParticleSystem<TDataType>> child)
	{
		return false;
	}
}