/*
 * @file driver_plugin_base.h 
 * @Basic class for plugins of a simulation driver.
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_DRIVER_DRIVER_PLUGIN_BASE_H_
#define PHYSIKA_DYNAMICS_DRIVER_DRIVER_PLUGIN_BASE_H_

namespace Physika{

template <typename Scalar> class DriverBase;

template <typename Scalar>
class DriverPluginBase
{
public:
    //constructors && deconstructors
    DriverPluginBase();
    virtual ~DriverPluginBase();

    //functions called in driver
    virtual void onInitialize(int frame) = 0;
    virtual void onBeginFrame(int frame) = 0;
    virtual void onEndFrame(int frame) = 0;
    virtual void onBeginTimeStep(Scalar dt) = 0;//replaced by onBeginRigidStep in rigid body simulation
    virtual void onEndTimeStep(Scalar time, Scalar dt) = 0;//replaced by onEndRigidStep in rigid body simulation
    virtual void onWrite(int frame) = 0;
    virtual void onRead(int frame) = 0;
    virtual void onRestart(int frame)=0;

    //basic function
    virtual DriverBase<Scalar>* driver();
    virtual void setDriver(DriverBase<Scalar>* driver) = 0;//should be redefined in child class because type-check of driver should be done before assignment.

protected:
    DriverBase<Scalar>* driver_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_DRIVER_DRIVER_PLUGIN_BASE_H_