/*
 * Copyright 2018 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * v1.0 Yukihiro Saito
 */

#include "multi_object_tracker/tracker/model/bicycle_tracker.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "multi_object_tracker/utils/utils.hpp"
#define EIGEN_MPL2_ONLY
#include <Eigen/Core>
#include <Eigen/Geometry>

BicycleTracker::BicycleTracker(const autoware_msgs::DynamicObject &object)
    : Tracker(object.semantic.type),
      filtered_posx_(object.state.pose.pose.position.x),
      filtered_posy_(object.state.pose.pose.position.y),
      filtered_vx_(0.0),
      filtered_vy_(0.0),
      v_filter_gain_(0.6),
      area_filter_gain_(0.8),
      prediction_time(ros::Time::now()),
      measurement_time(ros::Time::now())
{
    object_ = object;
    // area
    filtered_area_ = utils::getArea(object.shape);
}

bool BicycleTracker::predict(const ros::Time &time)
{
    double dt = (time - prediction_time).toSec();
    if (dt < 0.0)
        dt = 0.0;
    filtered_posx_ += filtered_vx_ * dt;
    filtered_posy_ += filtered_vy_ * dt;
    prediction_time = time;
    return true;
}
bool BicycleTracker::measure(const autoware_msgs::DynamicObject &object, const ros::Time &time)
{
    int type = object.semantic.type;
    bool is_changed_unknown_object = false;
    if (type == autoware_msgs::Semantic::UNKNOWN)
    {
        type = getType();
        is_changed_unknown_object = true;
    }
    object_ = object;
    setType(type);

    // area
    filtered_area_ = area_filter_gain_ * filtered_area_ + (1.0 - area_filter_gain_) * utils::getArea(object.shape);

    // vx,vy
    double dt = (time - measurement_time).toSec();
    measurement_time = time;
    if (0.0 < dt)
    {
        double current_vel =
            std::sqrt((object.state.pose.pose.position.x - filtered_posx_) * (object.state.pose.pose.position.x - filtered_posx_) + (object.state.pose.pose.position.y - filtered_posy_) * (object.state.pose.pose.position.y - filtered_posy_));
        const double max_vel = 30.0;  /* [m/s]*/
        const double vel_scale = std::min(max_vel, current_vel) / current_vel;
        if (is_changed_unknown_object)
        {
            filtered_vx_ = 0.9 * filtered_vx_ + (1.0 - 0.9) * ((object.state.pose.pose.position.x - filtered_posx_) / dt) * vel_scale;
            filtered_vy_ = 0.9 * filtered_vy_ + (1.0 - 0.9) * ((object.state.pose.pose.position.y - filtered_posy_) / dt) * vel_scale;
        }
        else
        {
            filtered_vx_ = v_filter_gain_ * filtered_vx_ + (1.0 - v_filter_gain_) * ((object.state.pose.pose.position.x - filtered_posx_) / dt)*vel_scale;
            filtered_vy_ = v_filter_gain_ * filtered_vy_ + (1.0 - v_filter_gain_) * ((object.state.pose.pose.position.y - filtered_posy_) / dt)*vel_scale;
            v_filter_gain_ = std::min(0.9, v_filter_gain_ + 0.05);
        }
    }


    // pos x, pos y
    filtered_posx_ = object.state.pose.pose.position.x;
    filtered_posy_ = object.state.pose.pose.position.y;
    // filtered_posx_ = pos_filter_gain_ * filtered_posx_ + (1.0 - pos_filter_gain_) * object.state.pose.pose.position.x;
    // filtered_posy_ = pos_filter_gain_ * filtered_posy_ + (1.0 - pos_filter_gain_) * object.state.pose.pose.position.y;

    return true;
}

bool BicycleTracker::getEstimatedDynamicObject(autoware_msgs::DynamicObject &object)
{
    object = object_;
    object.id = unique_id::toMsg(uuid_);
    object.semantic.type = getType();

    object.state.pose.pose.position.x = filtered_posx_;
    object.state.pose.pose.position.y = filtered_posy_;

    object.state.pose_reliable = false;

    double roll, pitch, yaw;
    tf2::Quaternion quaternion;
    tf2::fromMsg(object.state.pose.pose.orientation, quaternion);
    tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
    object.state.twist.twist.linear.x = filtered_vx_ * std::cos(-yaw) - filtered_vy_ * std::sin(-yaw);
    object.state.twist.twist.linear.y = filtered_vx_ * std::sin(-yaw) + filtered_vy_ * std::cos(-yaw);

    object.state.twist_reliable = true;

    return true;
}

geometry_msgs::Point BicycleTracker::getPosition()
{
    geometry_msgs::Point position;
    position.x = filtered_posx_;
    position.y = filtered_posy_;
    position.z = object_.state.pose.pose.position.z;
    return position;
}

double BicycleTracker::getArea()
{
    return filtered_area_;
}