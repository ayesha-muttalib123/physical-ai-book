---
id: 06-Chapter-2-Practical-ROS2-Examples
title: "Chapter 2: Practical ROS2 Examples"
sidebar_position: 6
---

# Chapter 2: Practical ROS2 Examples

## Overview

This chapter provides practical, real-world examples of ROS2 implementations in robotics applications. Students will work through complete examples that demonstrate how to combine all ROS2 communication patterns to build functional robotic systems. The chapter covers robot control, sensor integration, navigation, and system integration with hands-on exercises that reinforce the theoretical concepts learned in previous chapters.

## Why It Matters

Practical examples are essential for understanding how ROS2 concepts apply to real robotic systems. This chapter bridges the gap between theory and implementation, showing students how to combine nodes, topics, services, actions, and parameters to solve actual robotics problems. These examples provide templates and patterns that students can adapt for their own robotic applications.

## Key Concepts

### Complete ROS2 System Architecture
How all components work together in a real system. Understanding the integration of nodes, topics, services, actions, and parameters in a complete robotic application.

### Robot Bringup Procedures
Standard processes for starting robot systems. These procedures ensure all required nodes start in the correct order with proper configuration and parameter settings.

### Sensor Integration
Combining multiple sensors using appropriate ROS2 patterns. This involves selecting the right communication patterns for different sensor types and synchronizing data from multiple sources.

### Robot Control Systems
Implementing feedback control with ROS2. This includes creating control loops that process sensor data and generate appropriate commands to achieve desired robot behavior.

### Navigation Stacks
Understanding and implementing ROS2 navigation. The navigation stack provides modular components for localization, mapping, path planning, and motion control.

### System Debugging
Techniques for troubleshooting ROS2 systems. This includes using ROS2 tools for introspection, logging, and diagnosing communication issues.

### Performance Optimization
Best practices for efficient ROS2 implementations. This involves optimizing communication patterns, reducing latency, and managing resources effectively.

### Safety Considerations
Implementing safety in ROS2 robot systems. This includes creating safety checks, emergency stops, and fail-safe behaviors to ensure safe robot operation.

## Code Examples

### Complete Mobile Robot Controller

Integrated ROS2 node that controls a differential drive robot with sensor feedback:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math
import numpy as np

class MobileRobotController(Node):
    def __init__(self):
        super().__init__('mobile_robot_controller')

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)

        # Subscriptions
        self.odom_subscription = self.create_subscription(
            Odometry, 'odom', self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, sensor_qos)

        # Parameters
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('control_frequency', 10)

        # Robot state
        self.current_pose = None
        self.current_twist = None
        self.scan_data = None
        self.safety_enabled = True

        # Control timer
        control_freq = self.get_parameter('control_frequency').value
        self.control_timer = self.create_timer(1.0/control_freq, self.control_loop)

        self.get_logger().info('Mobile robot controller initialized')

    def odom_callback(self, msg):
        # Extract position and orientation from odometry
        self.current_pose = msg.pose.pose
        self.current_twist = msg.twist.twist

    def scan_callback(self, msg):
        # Store latest scan data
        self.scan_data = msg

    def control_loop(self):
        if self.scan_data is None or self.current_pose is None:
            return

        # Get parameters
        linear_speed = self.get_parameter('linear_speed').value
        angular_speed = self.get_parameter('angular_speed').value
        safety_dist = self.get_parameter('safety_distance').value

        # Check for obstacles
        if self.safety_enabled and self.has_obstacle_ahead(safety_dist):
            # Stop the robot if obstacle is detected
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.cmd_vel_publisher.publish(cmd_vel)
            self.get_logger().warn('Obstacle detected! Stopping robot.')
            return

        # Example: Simple wall following behavior
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_speed * 0.7  # Move forward at 70% speed
        cmd_vel.angular.z = self.calculate_wall_following_turn()

        self.cmd_vel_publisher.publish(cmd_vel)

    def has_obstacle_ahead(self, threshold):
        if self.scan_data is None:
            return False

        # Check the front 30 degrees for obstacles
        front_indices = range(
            len(self.scan_data.ranges) // 2 - 15,
            len(self.scan_data.ranges) // 2 + 15
        )

        for i in front_indices:
            if 0 < self.scan_data.ranges[i] < threshold:
                return True
        return False

    def calculate_wall_following_turn(self):
        if self.scan_data is None:
            return 0.0

        # Simple wall following: turn toward open space
        left_avg = np.mean(self.scan_data.ranges[:len(self.scan_data.ranges)//3])
        right_avg = np.mean(self.scan_data.ranges[2*len(self.scan_data.ranges)//3:])

        # Turn toward the side with more open space
        if left_avg > right_avg:
            return 0.3  # Turn right
        else:
            return -0.3  # Turn left

    def set_goal(self, x, y):
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0

        self.goal_publisher.publish(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = MobileRobotController()

    # Example: Set a goal after 5 seconds
    def set_example_goal():
        controller.set_goal(2.0, 1.0)

    goal_timer = controller.create_timer(5.0, set_example_goal)

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
```

### Multi-Sensor Fusion Node

Node that combines data from multiple sensors for enhanced perception:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
import numpy as np
from cv_bridge import CvBridge
import cv2

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Latest sensor data
        self.laser_data = None
        self.image_data = None
        self.imu_data = None

        # QoS profiles
        sensor_qos = rclpy.qos.QoSProfile(depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)

        # Subscriptions
        self.laser_subscription = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.image_subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, sensor_qos)
        self.imu_subscription = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Publishers
        self.fused_data_publisher = self.create_publisher(
            PoseWithCovarianceStamped, 'fused_pose', 10)
        self.status_publisher = self.create_publisher(String, 'fusion_status', 10)

        # Timer for fusion
        self.fusion_timer = self.create_timer(0.1, self.perform_fusion)

        self.get_logger().info('Sensor fusion node initialized')

    def laser_callback(self, msg):
        self.laser_data = msg
        self.get_logger().debug('Received laser data')

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_data = cv_image
            self.get_logger().debug('Received image data')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def imu_callback(self, msg):
        self.imu_data = msg
        self.get_logger().debug('Received IMU data')

    def perform_fusion(self):
        if not all([self.laser_data, self.image_data, self.imu_data]):
            status_msg = String()
            status_msg.data = 'Waiting for complete sensor data'
            self.status_publisher.publish(status_msg)
            return

        # Perform sensor fusion (simplified example)
        try:
            # Extract orientation from IMU
            orientation = self.extract_orientation_from_imu(self.imu_data)

            # Process laser data for position estimation
            position = self.estimate_position_from_laser(self.laser_data)

            # Combine data into fused estimate
            fused_pose = self.create_fused_estimate(position, orientation)

            # Publish fused data
            self.fused_data_publisher.publish(fused_pose)

            # Publish status
            status_msg = String()
            status_msg.data = f'Fusion completed: pos=({position[0]:.2f},{position[1]:.2f}), orient={orientation[2]:.2f}'
            self.status_publisher.publish(status_msg)

            self.get_logger().info(status_msg.data)

        except Exception as e:
            self.get_logger().error(f'Fusion error: {e}')

    def extract_orientation_from_imu(self, imu_msg):
        # Extract orientation from IMU quaternion
        import math
        x, y, z, w = (imu_msg.orientation.x, imu_msg.orientation.y,
                      imu_msg.orientation.z, imu_msg.orientation.w)

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return (roll, pitch, yaw)

    def estimate_position_from_laser(self, laser_msg):
        # Simple position estimation from laser scan (could be more sophisticated)
        # For this example, we'll just use a basic approach
        ranges = [r for r in laser_msg.ranges if not (r == float('inf') or r == float('nan'))]

        if ranges:
            # Use the median range as a simple position estimate
            median_range = np.median(ranges)
            # This is a simplified approach - in reality, you'd use more sophisticated methods
            return (median_range * 0.7, 0.0)  # x, y position estimate
        else:
            return (0.0, 0.0)

    def create_fused_estimate(self, position, orientation):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        # Set position
        pose_msg.pose.pose.position.x = position[0]
        pose_msg.pose.pose.position.y = position[1]
        pose_msg.pose.pose.position.z = 0.0

        # Set orientation (simplified - convert Euler to quaternion)
        yaw = orientation[2]
        pose_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Set covariance (simplified)
        pose_msg.pose.covariance = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.1]

        return pose_msg

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusionNode()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()
```

### ROS2 Navigation Stack Integration

Example of integrating with ROS2 navigation stack components:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import String
from rclpy.duration import Duration

# Import navigation action (would use actual nav2_msgs in real implementation)
# from nav2_msgs.action import NavigateToPose

class NavigationIntegrationNode(Node):
    def __init__(self):
        super().__init__('navigation_integration')

        # Publishers
        self.goal_publisher = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.path_subscriber = self.create_subscription(
            Path, 'plan', self.path_callback, 10)
        self.status_publisher = self.create_publisher(String, 'nav_status', 10)

        # Action client for navigation (using placeholder since nav2_msgs may not be available)
        # self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timer to send navigation goals
        self.nav_timer = self.create_timer(10.0, self.send_navigation_goal)

        self.get_logger().info('Navigation integration node initialized')

    def path_callback(self, msg):
        # Handle received navigation path
        self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')

    def send_navigation_goal(self):
        # Create and send a navigation goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'

        # Set goal position (example: move 2m forward)
        goal_msg.pose.position.x = 2.0
        goal_msg.pose.position.y = 1.0
        goal_msg.pose.position.z = 0.0

        # Set goal orientation (facing forward)
        goal_msg.pose.orientation.w = 1.0

        self.goal_publisher.publish(goal_msg)
        self.get_logger().info(f'Sent navigation goal to ({goal_msg.pose.position.x}, {goal_msg.pose.position.y})')

        # Publish status
        status_msg = String()
        status_msg.data = f'Navigation goal sent to ({goal_msg.pose.position.x}, {goal_msg.pose.position.y})'
        self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    nav_node = NavigationIntegrationNode()

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        pass
    finally:
        nav_node.destroy_node()
        rclpy.shutdown()
```

## Practical Examples

### Autonomous Patrol Robot

Students implement a complete autonomous patrol robot that navigates between waypoints, avoids obstacles, and reports status.

**Objectives:**
- Integrate sensor processing with navigation
- Implement obstacle avoidance behavior
- Create patrol path following
- Implement status reporting

**Required Components:**
- Laser scanner for obstacle detection
- Odometry for localization
- Velocity command interface
- Navigation system

**Evaluation Criteria:**
- Successful waypoint navigation
- Effective obstacle avoidance
- Robust patrol behavior
- Reliable status reporting

### Object Detection and Grasping System

Students develop a system that detects objects using camera and LIDAR, then plans and executes grasping motions.

**Objectives:**
- Integrate visual and range sensing
- Implement object detection and localization
- Plan grasping motions
- Execute grasping with feedback

**Required Components:**
- Camera for object detection
- LIDAR for range sensing
- Robotic arm with gripper
- Motion planning system

**Evaluation Criteria:**
- Accurate object detection
- Successful grasping attempts
- Robust sensor integration
- Safe motion execution

### Multi-Robot Coordination System

Students create a system where multiple robots coordinate to perform a task, sharing information and avoiding conflicts.

**Objectives:**
- Implement inter-robot communication
- Coordinate robot activities
- Avoid resource conflicts
- Share environmental information

**Required Components:**
- Multiple robot platforms
- Communication network
- Task allocation system
- Conflict resolution

**Evaluation Criteria:**
- Effective coordination
- Conflict-free operation
- Efficient resource usage
- Robust communication

## Summary

Chapter 5 provides practical, real-world examples of ROS2 implementations in robotics applications. Students learned how to combine all ROS2 communication patterns to build functional robotic systems including mobile robot controllers, sensor fusion systems, and navigation integration. The examples demonstrated complete system architectures and provided templates for students to adapt for their own applications.

## Quiz

1. What is the primary purpose of a robot bringup procedure?
   - A: To physically build the robot
   - B: Standard processes for starting robot systems with all required nodes
   - C: To calibrate sensors
   - D: To write code for the robot

   **Answer: B** - Robot bringup procedures are standard processes for starting robot systems with all required nodes in the correct order and with proper configuration.

2. Why is sensor fusion important in robotics?
   - A: It makes sensors cheaper
   - B: It combines data from multiple sensors for enhanced perception and reliability
   - C: It reduces the number of sensors needed
   - D: It makes sensors faster

   **Answer: B** - Sensor fusion combines data from multiple sensors to create more accurate, reliable, and comprehensive understanding of the environment than any single sensor could provide.

3. What QoS policy should be used for critical safety-related messages?
   - A: Best effort with small history
   - B: Reliable with keep-all history
   - C: Volatile durability
   - D: Deadline-based policy only

   **Answer: B** - For critical safety-related messages, reliable delivery with keep-all history ensures that all messages are delivered and none are lost.

4. What is the main benefit of using ROS2 navigation stack?
   - A: It makes robots move faster
   - B: It provides tested, modular components for robot navigation
   - C: It reduces hardware requirements
   - D: It eliminates the need for sensors

   **Answer: B** - The ROS2 navigation stack provides tested, modular components for robot navigation that can be configured and used rather than implementing navigation from scratch.

5. When should you use composition in ROS2?
   - A: Always, to make systems faster
   - B: To run multiple nodes in the same process for efficiency and reduced latency
   - C: Only for debugging
   - D: When you need more memory

   **Answer: B** - Composition is used to run multiple nodes in the same process to improve efficiency and reduce latency by avoiding network overhead.

## Learning Outcomes

After completing this chapter, students will be able to:
- Design ROS2 architectures for robot systems
- Implement nodes, topics, services, and actions
- Manage parameters and configurations
- Develop ROS2 packages for multi-robot systems

## Prerequisites

- Basic understanding of Python programming
- Fundamentals of linear algebra and calculus
- Basic knowledge of robotics concepts
- Introduction to machine learning concepts
- Completion of Module 0 (Introduction and Foundations)
- Completion of Chapter 01 (Physical AI Basics)
- Completion of Chapter 03 (ROS2 Nodes, Topics, and Services)
- Completion of Chapter 04 (ROS2 Communication Patterns)

## Estimated Duration

6 hours