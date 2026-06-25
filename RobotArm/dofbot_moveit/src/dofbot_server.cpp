#include <iostream>
#include <vector>
#include <cmath>
#include <rclcpp/rclcpp.hpp>

#include "dofbot_kinemarics.h"
#include "dofbot_info/srv/kinemarics.hpp"

using namespace KDL;
using namespace std;

Dofbot dofbot = Dofbot();
// 弧度转角度
const float RA2DE = 180.0f / M_PI;
// 角度转弧度
const float DE2RA = M_PI / 180.0f;
int a = 0;

class KinematicsNode : public rclcpp::Node {
public:
    KinematicsNode() : Node("dofbot_kinemarics_server") {
        // 完全匹配截图里的真实路径
        urdf_path_ = "/home/HwHiAiUser/E2Esamples/src/E2E-Sample/ros2_robot_arm/ros2_ws/src/dofbot_moveit/urdf/dofbot.urdf";
        // 创建运动学服务 trial_service
        service_ = this->create_service<dofbot_info::srv::Kinemarics>(
            "trial_service",
            std::bind(&KinematicsNode::srvicecallback, this, std::placeholders::_1, std::placeholders::_2)
        );
        RCLCPP_INFO(this->get_logger(), "ROS Server Starts..");
    }

private:
    bool srvicecallback(
        const std::shared_ptr<dofbot_info::srv::Kinemarics::Request> request,
        std::shared_ptr<dofbot_info::srv::Kinemarics::Response> response
    ) {
        const char *urdf_file = urdf_path_.c_str();
        // 正运动学 FK
        if (request->kin_name == "fk") {
            double joints[]{
                request->cur_joint1,
                request->cur_joint2,
                request->cur_joint3,
                request->cur_joint4,
                request->cur_joint5
            };
            vector<double> initjoints;
            vector<double> initpos;
            for (int i = 0; i < 5; ++i) {
                initjoints.push_back((joints[i] - 90) * DE2RA);
            }
            dofbot.dofbot_getFK(urdf_file, initjoints, initpos);
            cout << "--------- Fk ---------" << a << "--------- Fk ---------" << endl;
            cout << "XYZ坐标 ： " << initpos.at(0) << " ," << initpos.at(1) << " ," << initpos.at(2) << endl;
            cout << "Roll,Pitch,Yaw： " << initpos.at(3) << " ," << initpos.at(4) << " ," << initpos.at(5) << endl;
            response->x = initpos.at(0);
            response->y = initpos.at(1);
            response->z = initpos.at(2);
            response->roll = initpos.at(3);
            response->pitch = initpos.at(4);
            response->yaw = initpos.at(5);
        }

        // 逆运动学 IK
        if (request->kin_name == "ik") {
            double tool_param = 0.12;
            double Roll = 2.5 * request->tar_y * 100 - 207.5;
            double Pitch = 0;
            double Yaw = 0;
            double init_angle = atan2(double(request->tar_x), double(request->tar_y));
            double dist = tool_param * sin((180 + Roll) * DE2RA);
            double distance = hypot(request->tar_x, request->tar_y) - dist;
            double x = distance * sin(init_angle);
            double y = distance * cos(init_angle);
            double z = tool_param * cos((180 + Roll) * DE2RA);

            if (request->tar_z >= 0.2) {
                x = request->tar_x;
                y = request->tar_y;
                z = request->tar_z;
                Roll = -90;
            }

            double xyz[]{x, y, z};
            double rpy[]{Roll * DE2RA, Pitch * DE2RA, Yaw * DE2RA};
            cout << xyz[0] << " , " << xyz[1] << " , " << xyz[2] << "\t"
                 << rpy[0] << " , " << rpy[1] << " , " << rpy[2] << endl;

            vector<double> outjoints;
            vector<double> targetXYZ;
            vector<double> targetRPY;
            for (int k = 0; k < 3; ++k) targetXYZ.push_back(xyz[k]);
            for (int l = 0; l < 3; ++l) targetRPY.push_back(rpy[l]);

            dofbot.dofbot_getIK(urdf_file, targetXYZ, targetRPY, outjoints);
            cout << "--------- Ik ---------" << a << "--------- Ik ---------" << endl;
            for (int i = 0; i < 5; i++) {
                cout << (outjoints.at(i) * RA2DE) + 90 << ",";
            }
            a++;
            response->joint1 = (outjoints.at(0) * RA2DE) + 90;
            response->joint2 = (outjoints.at(1) * RA2DE) + 90;
            response->joint3 = (outjoints.at(2) * RA2DE) + 90;
            response->joint4 = (outjoints.at(3) * RA2DE) + 90;
            response->joint5 = (outjoints.at(4) * RA2DE) + 90;
            cout << "\nFinish adding to response, and joint1 is: " << response->joint1 << endl;
        }
        return true;
    }

    rclcpp::Service<dofbot_info::srv::Kinemarics>::SharedPtr service_;
    std::string urdf_path_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KinematicsNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}