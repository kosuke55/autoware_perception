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

#include "multi_object_tracker/data_association/data_association.hpp"
#include "hungarian/hungarian.h"
#include "multi_object_tracker/utils/utils.hpp"

// #include <iostream>
DataAssociation::DataAssociation()
    : score_threshold_(0.1)
{
    can_assgin_matrix_ = Eigen::MatrixXi::Identity(20, 20);
    can_assgin_matrix_(autoware_msgs::Semantic::UNKNOWN, autoware_msgs::Semantic::UNKNOWN) = 0;
    can_assgin_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::UNKNOWN) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::TRUCK) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::BUS) = 0;
    can_assgin_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::UNKNOWN) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::CAR) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::BUS) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::UNKNOWN) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::CAR) = 0;
    can_assgin_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::TRUCK) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::UNKNOWN) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::MOTORBIKE) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::UNKNOWN) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::BICYCLE) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::PEDESTRIAN, autoware_msgs::Semantic::UNKNOWN) = 1;
    can_assgin_matrix_(autoware_msgs::Semantic::ANIMAL, autoware_msgs::Semantic::UNKNOWN) = 1;
    max_dist_matrix_ = Eigen::MatrixXd::Constant(20, 20, 1.0);
    max_dist_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::UNKNOWN) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::CAR) = 4.5;
    max_dist_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::TRUCK) = 4.5;
    max_dist_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::BUS) = 4.5;
    max_dist_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::UNKNOWN) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::CAR) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::TRUCK) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::BUS) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::UNKNOWN) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::CAR) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::TRUCK) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::BUS) = 4.0;
    max_dist_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::UNKNOWN) = 3.0;
    max_dist_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::BICYCLE) = 3.0;
    max_dist_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::MOTORBIKE) = 3.0;
    max_dist_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::UNKNOWN) = 3.0;
    max_dist_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::BICYCLE) = 3.0;
    max_dist_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::MOTORBIKE) = 2.0;
    max_dist_matrix_(autoware_msgs::Semantic::PEDESTRIAN, autoware_msgs::Semantic::UNKNOWN) = 2.0;
    max_dist_matrix_(autoware_msgs::Semantic::PEDESTRIAN, autoware_msgs::Semantic::PEDESTRIAN) = 2.0;
    max_dist_matrix_(autoware_msgs::Semantic::ANIMAL, autoware_msgs::Semantic::UNKNOWN) = 1.0;
    max_area_matrix_ = Eigen::MatrixXd::Constant(20, 20, /* large number */ 10000.0);
    max_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::UNKNOWN) = 2.2 * 5.5;
    max_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::CAR) = 2.2 * 5.5;
    max_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::TRUCK) = 2.5 * 7.9;
    max_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::BUS) = 2.7 * 12.0;
    max_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::UNKNOWN) = 2.5 * 7.9;
    max_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::CAR) = 2.2 * 5.5;
    max_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::TRUCK) = 2.5 * 7.9;
    max_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::BUS) = 2.7 * 12.0;
    max_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::UNKNOWN) = 2.7 * 12.0;
    max_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::CAR) = 2.2 * 5.5;
    max_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::TRUCK) = 2.5 * 7.9;
    max_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::BUS) = 2.7 * 12.0;
    max_area_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::UNKNOWN) = 2.5;
    max_area_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::BICYCLE) = 2.5;
    max_area_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::MOTORBIKE) = 3.0;
    max_area_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::UNKNOWN) = 3.0;
    max_area_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::BICYCLE) = 2.5;
    max_area_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::MOTORBIKE) = 3.0;
    max_area_matrix_(autoware_msgs::Semantic::PEDESTRIAN, autoware_msgs::Semantic::UNKNOWN) = 2.0;
    max_area_matrix_(autoware_msgs::Semantic::PEDESTRIAN, autoware_msgs::Semantic::PEDESTRIAN) = 2.0;
    max_area_matrix_(autoware_msgs::Semantic::ANIMAL, autoware_msgs::Semantic::UNKNOWN) = 2.0;
    min_area_matrix_ = Eigen::MatrixXd::Constant(20, 20, /* small number */ 0.0);
    min_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::UNKNOWN) = 1.2 * 3.0;
    min_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::CAR) = 1.2 * 3.0;
    min_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::TRUCK) = 1.5 * 4.0;
    min_area_matrix_(autoware_msgs::Semantic::CAR, autoware_msgs::Semantic::BUS) = 2.0 * 5.0;
    min_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::UNKNOWN) = 1.5 * 4.0;
    min_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::CAR) = 1.2 * 3.0;
    min_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::TRUCK) = 1.5 * 4.0;
    min_area_matrix_(autoware_msgs::Semantic::TRUCK, autoware_msgs::Semantic::BUS) = 2.0 * 5.0;
    min_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::UNKNOWN) = 2.0 * 5.0;
    min_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::CAR) = 1.2 * 3.0;
    min_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::TRUCK) = 1.5 * 4.0;
    min_area_matrix_(autoware_msgs::Semantic::BUS, autoware_msgs::Semantic::BUS) = 2.0 * 5.0;
    min_area_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::UNKNOWN) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::BICYCLE) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::BICYCLE, autoware_msgs::Semantic::MOTORBIKE) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::UNKNOWN) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::BICYCLE) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::MOTORBIKE, autoware_msgs::Semantic::MOTORBIKE) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::PEDESTRIAN, autoware_msgs::Semantic::UNKNOWN) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::PEDESTRIAN, autoware_msgs::Semantic::PEDESTRIAN) = 0.001;
    min_area_matrix_(autoware_msgs::Semantic::ANIMAL, autoware_msgs::Semantic::UNKNOWN) = 0.5;
}

bool DataAssociation::assign(const Eigen::MatrixXd &src,
                             std::unordered_map<int, int> &direct_assignment,
                             std::unordered_map<int, int> &reverse_assignment)
{
    std::vector<std::vector<double>> score(src.rows());
    for (int row = 0; row < src.rows(); ++row)
    {
        score.at(row).resize(src.cols());
        for (int col = 0; col < src.cols(); ++col)
        {
            score.at(row).at(col) = src(row, col);
        }
    }
    operations_research::MaximizeLinearAssignment(score, &direct_assignment, &reverse_assignment);
    for (auto itr = direct_assignment.begin(); itr != direct_assignment.end();)
    {
        if (src(itr->first, itr->second) < score_threshold_)
        {
            itr = direct_assignment.erase(itr);
            continue;
        }
        else
        {
            ++itr;
        }
    }
    for (auto itr = reverse_assignment.begin(); itr != reverse_assignment.end();)
    {
        if (src(itr->second, itr->first) < score_threshold_)
        {
            itr = reverse_assignment.erase(itr);
            continue;
        }
        else
        {
            ++itr;
        }
    }
}

Eigen::MatrixXd DataAssociation::calcScoreMatrix(const autoware_msgs::DynamicObjectWithFeatureArray &measurements,
                                                 const std::list<std::shared_ptr<Tracker>> &trackers)
{
    Eigen::MatrixXd score_matrix = Eigen::MatrixXd::Zero(trackers.size(), measurements.feature_objects.size());
    size_t tracker_idx = 0;
    for (auto tracker_itr = trackers.begin(); tracker_itr != trackers.end(); ++tracker_itr, ++tracker_idx)
    {
        for (size_t measurement_idx = 0; measurement_idx < measurements.feature_objects.size(); ++measurement_idx)
        {
            double score = 0.0;
            if (can_assgin_matrix_((*tracker_itr)->getType(), measurements.feature_objects.at(measurement_idx).object.semantic.type))
            {
                double max_dist = max_dist_matrix_((*tracker_itr)->getType(), measurements.feature_objects.at(measurement_idx).object.semantic.type);
                double max_area = max_area_matrix_((*tracker_itr)->getType(), measurements.feature_objects.at(measurement_idx).object.semantic.type);
                double min_area = min_area_matrix_((*tracker_itr)->getType(), measurements.feature_objects.at(measurement_idx).object.semantic.type);
                double dist = getDistance(measurements.feature_objects.at(measurement_idx).object.state.pose.pose.position, (*tracker_itr)->getPosition());
                double area = utils::getArea(measurements.feature_objects.at(measurement_idx).object.shape);
                score = (max_dist - std::min(dist, max_dist)) / max_dist;

                if (max_dist < dist)
                    score = 0.0;
                if (area < min_area || max_area < area)
                    score = 0.0;
                // if ((*tracker_itr)->getType() == measurements.feature_objects.at(measurement_idx).object.semantic.type &&
                //     measurements.feature_objects.at(measurement_idx).object.semantic.type != autoware_msgs::Semantic::UNKNOWN)
                //     score += 1.0;
                // if (measurements.feature_objects.at(measurement_idx).object.semantic.type != autoware_msgs::Semantic::UNKNOWN)
                //     score += 1.0;
            }
            score_matrix(tracker_idx, measurement_idx) = score;
        }
    }

    return score_matrix;
}

double DataAssociation::getDistance(const geometry_msgs::Point &measurement,
                                    const geometry_msgs::Point &tracker)
{
    const double diff_x = tracker.x - measurement.x;
    const double diff_y = tracker.y - measurement.y;
    // const double diff_z = tracker.z - measurement.z;
    return std::sqrt(diff_x * diff_x + diff_y * diff_y);
}
