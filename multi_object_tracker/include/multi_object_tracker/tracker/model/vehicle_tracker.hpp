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
 * v1.0 Yukihiro Saito
 */

#pragma once
#include "tracker_base.hpp"
#include "autoware_msgs/DynamicObject.h"

class VehicleTracker : public Tracker
{
private:
  autoware_msgs::DynamicObject object_;
  double filtered_yaw_;
  double yaw_filter_gain_;
  bool is_fixed_yaw_;
  double filtered_dim_x_;
  double filtered_dim_y_;
  double dim_filter_gain_;
  bool is_fixed_dim_;
  double filtered_posx_;
  double filtered_posy_;
  double pos_filter_gain_;
  double filtered_vx_;
  double filtered_vy_;
  double v_filter_gain_;
  double filtered_area_;
  double area_filter_gain_;
  ros::Time prediction_time;
  ros::Time measurement_time;

public:
  VehicleTracker(const autoware_msgs::DynamicObject &object);

  bool predict(const ros::Time &time) override;
  bool measure(const autoware_msgs::DynamicObject &object, const ros::Time &time) override;
  bool getEstimatedDynamicObject(autoware_msgs::DynamicObject &object) override;
  geometry_msgs::Point getPosition() override;
  double getArea() override;
  virtual ~VehicleTracker(){};
};
