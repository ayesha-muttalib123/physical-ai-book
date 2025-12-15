---
id: 10-Chapter-2-Digital-Twin-Robotics-Examples
title: "Chapter 2: Digital Twin Robotics Examples"
sidebar_position: 10
---

# Chapter 2: Digital Twin Robotics Examples

## Overview

This chapter presents comprehensive examples of digital twin implementations using both Gazebo and Unity. Students will work through real-world scenarios that demonstrate the integration of physical and virtual worlds, including industrial automation, autonomous vehicles, and robotic manipulation tasks. Each example showcases best practices for connecting simulation to real hardware and leveraging digital twins for enhanced robot development and deployment.

## Why It Matters

Real-world examples provide essential context for understanding how digital twins are implemented in practice. These examples demonstrate the practical benefits of digital twin technology in robotics, including reduced development time, improved safety, and enhanced testing capabilities. By working through these examples, students gain hands-on experience with the challenges and solutions involved in creating effective digital twin systems.

## Key Concepts

### Industrial Automation Digital Twins
Creating digital replicas of manufacturing environments. These digital twins enable the simulation of complex manufacturing processes, allowing for optimization and validation before deployment on actual production lines.

### Autonomous Vehicle Simulation
Combining sensor simulation with real-world validation. This involves creating realistic sensor models and environmental conditions that accurately reflect real-world scenarios for testing autonomous vehicles.

### Robotic Manipulation Twins
Bridging simulation and reality for dexterous tasks. These digital twins focus on precise manipulation tasks, incorporating force feedback and haptic interfaces to enhance the realism of the simulation.

### Data Synchronization
Aligning real sensor data with simulated environments. This involves maintaining temporal and spatial consistency between real and simulated systems to ensure accurate representation.

### Hardware-in-the-Loop Testing
Integrating real hardware with simulated environments. This approach combines physical components with virtual simulation to validate system behavior in realistic conditions.

### Performance Validation
Comparing real vs. simulated robot behaviors. This involves developing metrics and methodologies to assess the accuracy and effectiveness of digital twin implementations.

### Fleet Management Twins
Digital twins for multi-robot systems. These digital twins manage and coordinate multiple robots, optimizing fleet performance and resource allocation.

### Predictive Maintenance
Using digital twins for equipment monitoring and maintenance. Digital twins can monitor equipment health and predict maintenance needs, reducing downtime and extending equipment life.

## Code Examples

### Industrial Robot Arm Digital Twin

Implementation of a digital twin for an industrial robot arm with real-time synchronization:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
import time

class IndustrialArmDigitalTwin(Node):
    def __init__(self):
        super().__init__('industrial_arm_digital_twin')

        # Real robot joint state subscriber
        self.real_joint_sub = self.create_subscription(
            JointState,
            '/real_robot/joint_states',
            self.real_joint_callback,
            10
        )

        # Simulated robot joint command publisher
        self.sim_cmd_pub = self.create_publisher(
            JointTrajectory,
            '/sim_robot/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Simulated robot joint state subscriber
        self.sim_joint_sub = self.create_subscription(
            JointState,
            '/sim_robot/joint_states',
            self.sim_joint_callback,
            10
        )

        # Real robot command publisher
        self.real_cmd_pub = self.create_publisher(
            JointTrajectory,
            '/real_robot/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Timer for synchronization
        self.timer = self.create_timer(0.1, self.synchronization_callback)  # 10Hz

        # State storage
        self.real_joints = {}
        self.sim_joints = {}
        self.sync_enabled = True

    def real_joint_callback(self, msg):
        """Receive joint states from the real robot"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.real_joints[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0,
                    'timestamp': self.get_clock().now().to_msg()
                }

    def sim_joint_callback(self, msg):
        """Receive joint states from the simulated robot"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.sim_joints[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0,
                    'timestamp': self.get_clock().now().to_msg()
                }

    def synchronization_callback(self):
        """Synchronize real and simulated robot states"""
        if not self.sync_enabled:
            return

        # Option 1: Send real robot state to simulation (for visualization)
        if self.should_update_simulation():
            self.update_simulation_from_real()

        # Option 2: Send simulated commands to real robot (for validation)
        if self.should_update_real_robot():
            self.update_real_robot_from_simulation()

    def should_update_simulation(self):
        """Determine if simulation should be updated from real robot"""
        # In real applications, this could be based on various criteria
        return True

    def should_update_real_robot(self):
        """Determine if real robot should be updated from simulation"""
        # Only update real robot during testing or validation phases
        return False  # Disabled by default for safety

    def update_simulation_from_real(self):
        """Send real robot state to simulation for visualization"""
        if not self.real_joints:
            return

        # Create trajectory message with current real positions
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = list(self.real_joints.keys())

        point = JointTrajectoryPoint()
        for joint_name in traj_msg.joint_names:
            joint_data = self.real_joints[joint_name]
            point.positions.append(joint_data['position'])
            point.velocities.append(joint_data['velocity'])
            point.effort.append(joint_data['effort'])

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1 seconds

        traj_msg.points.append(point)
        self.sim_cmd_pub.publish(traj_msg)

    def update_real_robot_from_simulation(self):
        """Send simulated commands to real robot for validation"""
        if not self.sim_joints:
            return

        # In a real implementation, this would be much more sophisticated
        # and include safety checks and validation
        traj_msg = JointTrajectory()
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        traj_msg.joint_names = list(self.sim_joints.keys())

        point = JointTrajectoryPoint()
        for joint_name in traj_msg.joint_names:
            joint_data = self.sim_joints[joint_name]
            point.positions.append(joint_data['position'])
            point.velocities.append(joint_data['velocity'])
            point.effort.append(joint_data['effort'])

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 0.1 seconds

        traj_msg.points.append(point)
        self.real_cmd_pub.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IndustrialArmDigitalTwin()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down industrial arm digital twin...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Autonomous Mobile Robot Fleet Digital Twin

Digital twin implementation for a fleet of mobile robots with real-time tracking and simulation:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import math
from collections import defaultdict

class MobileRobotFleetDigitalTwin(Node):
    def __init__(self):
        super().__init__('mobile_fleet_digital_twin')

        # Fleet state storage
        self.robot_states = defaultdict(dict)
        self.robot_goals = defaultdict(None)
        self.fleet_metrics = {
            'total_distance_traveled': defaultdict(float),
            'total_operational_time': defaultdict(0.0),
            'average_speed': defaultdict(float),
            'collision_count': defaultdict(int),
            'task_completion_rate': defaultdict(float)
        }

        # Publishers and subscribers
        self.odom_subs = {}
        self.cmd_vel_pubs = {}
        self.sim_odom_subs = {}
        self.sim_cmd_pubs = {}
        self.visualization_pub = self.create_publisher(MarkerArray, '/fleet_visualization', 10)
        self.metrics_pub = self.create_publisher(Float32, '/fleet_metrics', 10)

        # Timer for periodic tasks
        self.timer = self.create_timer(0.5, self.periodic_tasks)

        # Fleet configuration
        self.fleet_size = 5
        self.robot_names = [f'robot_{i}' for i in range(self.fleet_size)]
        self.initialize_fleet()

    def initialize_fleet(self):
        """Initialize subscribers and publishers for each robot"""
        for robot_name in self.robot_names:
            # Real robot subscribers
            self.odom_subs[robot_name] = self.create_subscription(
                Odometry,
                f'/{robot_name}/odometry/filtered',
                lambda msg, rn=robot_name: self.real_odom_callback(msg, rn),
                10
            )

            # Real robot publishers
            self.cmd_vel_pubs[robot_name] = self.create_publisher(
                Twist,
                f'/{robot_name}/cmd_vel',
                10
            )

            # Simulated robot subscribers
            self.sim_odom_subs[robot_name] = self.create_subscription(
                Odometry,
                f'/sim_{robot_name}/odometry/filtered',
                lambda msg, rn=robot_name: self.sim_odom_callback(msg, rn),
                10
            )

            # Simulated robot publishers
            self.sim_cmd_pubs[robot_name] = self.create_publisher(
                Twist,
                f'/sim_{robot_name}/cmd_vel',
                10
            )

    def real_odom_callback(self, msg, robot_name):
        """Process odometry from real robot"""
        self.robot_states[robot_name]['real'] = {
            'pose': msg.pose.pose,
            'twist': msg.twist.twist,
            'timestamp': msg.header.stamp
        }

        # Update fleet metrics
        self.update_metrics(robot_name, 'real', msg)

    def sim_odom_callback(self, msg, robot_name):
        """Process odometry from simulated robot"""
        self.robot_states[robot_name]['sim'] = {
            'pose': msg.pose.pose,
            'twist': msg.twist.twist,
            'timestamp': msg.header.stamp
        }

        # Update fleet metrics
        self.update_metrics(robot_name, 'sim', msg)

    def update_metrics(self, robot_name, source, odom_msg):
        """Update fleet metrics based on robot data"""
        if source == 'real':
            # Update real robot metrics
            pose = odom_msg.pose.pose
            twist = odom_msg.twist.twist

            # Calculate distance traveled incrementally
            if f'last_pose_{source}' in self.robot_states[robot_name]:
                last_pose = self.robot_states[robot_name][f'last_pose_{source}']
                distance = self.calculate_distance(pose, last_pose)
                self.fleet_metrics['total_distance_traveled'][robot_name] += distance

            self.robot_states[robot_name][f'last_pose_{source}'] = pose
            self.fleet_metrics['average_speed'][robot_name] = twist.linear.x

    def calculate_distance(self, pose1, pose2):
        """Calculate Euclidean distance between two poses"""
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        dz = pose1.position.z - pose2.position.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def periodic_tasks(self):
        """Execute periodic tasks for fleet management"""
        # Update visualization
        self.update_visualization()

        # Compare real vs simulated performance
        self.compare_real_sim_performance()

        # Check for anomalies
        self.check_anomalies()

    def update_visualization(self):
        """Update fleet visualization markers"""
        marker_array = MarkerArray()

        for i, robot_name in enumerate(self.robot_names):
            if 'real' in self.robot_states[robot_name]:
                # Real robot marker
                real_marker = Marker()
                real_marker.header.frame_id = "map"
                real_marker.header.stamp = self.get_clock().now().to_msg()
                real_marker.ns = "real_robots"
                real_marker.id = i * 2
                real_marker.type = Marker.CYLINDER
                real_marker.action = Marker.ADD

                real_pose = self.robot_states[robot_name]['real']['pose']
                real_marker.pose.position = real_pose.position
                real_marker.pose.orientation = real_pose.orientation

                real_marker.scale.x = 0.5
                real_marker.scale.y = 0.5
                real_marker.scale.z = 0.2

                real_marker.color.a = 0.8  # Alpha
                real_marker.color.r = 1.0  # Red
                real_marker.color.g = 0.0
                real_marker.color.b = 0.0

                marker_array.markers.append(real_marker)

            if 'sim' in self.robot_states[robot_name]:
                # Simulated robot marker
                sim_marker = Marker()
                sim_marker.header.frame_id = "map"
                sim_marker.header.stamp = self.get_clock().now().to_msg()
                sim_marker.ns = "sim_robots"
                sim_marker.id = i * 2 + 1
                sim_marker.type = Marker.CUBE
                sim_marker.action = Marker.ADD

                sim_pose = self.robot_states[robot_name]['sim']['pose']
                sim_marker.pose.position = sim_pose.position
                sim_marker.pose.orientation = sim_pose.orientation

                sim_marker.scale.x = 0.5
                sim_marker.scale.y = 0.5
                sim_marker.scale.z = 0.2

                sim_marker.color.a = 0.6  # Alpha
                sim_marker.color.r = 0.0  # Blue
                sim_marker.color.g = 0.0
                sim_marker.color.b = 1.0

                marker_array.markers.append(sim_marker)

        self.visualization_pub.publish(marker_array)

    def compare_real_sim_performance(self):
        """Compare real and simulated robot performance"""
        for robot_name in self.robot_names:
            if 'real' in self.robot_states[robot_name] and 'sim' in self.robot_states[robot_name]:
                real_pose = self.robot_states[robot_name]['real']['pose']
                sim_pose = self.robot_states[robot_name]['sim']['pose']

                # Calculate position difference
                pos_diff = self.calculate_distance(real_pose, sim_pose)

                # Log differences for analysis
                if pos_diff > 0.5:  # Threshold for significant difference
                    self.get_logger().warn(f'Large position difference for {robot_name}: {pos_diff:.2f}m')

    def check_anomalies(self):
        """Check for anomalies in fleet behavior"""
        for robot_name in self.robot_names:
            if 'real' in self.robot_states[robot_name]:
                # Check for collision indicators (abrupt velocity changes)
                twist = self.robot_states[robot_name]['real']['twist']
                speed = math.sqrt(twist.linear.x**2 + twist.linear.y**2)

                if hasattr(self, f'last_speed_{robot_name}'):
                    speed_change = abs(speed - getattr(self, f'last_speed_{robot_name}'))
                    if speed_change > 2.0:  # Significant acceleration
                        self.get_logger().info(f'Potential collision detected for {robot_name}')
                        self.fleet_metrics['collision_count'][robot_name] += 1

                setattr(self, f'last_speed_{robot_name}', speed)

def main(args=None):
    rclpy.init(args=args)
    node = MobileRobotFleetDigitalTwin()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down fleet digital twin...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Robotic Manipulation Digital Twin with Force Feedback

Digital twin for robotic manipulation with haptic feedback and force sensing:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, WrenchStamped
from geometry_msgs.msg import Wrench, PoseStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import math

class ManipulationDigitalTwin(Node):
    def __init__(self):
        super().__init__('manipulation_digital_twin')

        # Publishers and subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/real_manipulator/joint_states',
            self.joint_state_callback,
            10
        )

        self.force_torque_sub = self.create_subscription(
            WrenchStamped,
            '/real_manipulator/wrench',
            self.force_torque_callback,
            10
        )

        self.end_effector_pose_sub = self.create_subscription(
            PoseStamped,
            '/real_manipulator/end_effector_pose',
            self.end_effector_pose_callback,
            10
        )

        self.sim_joint_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/sim_manipulator/joint_group_position_controller/commands',
            10
        )

        self.haptic_feedback_pub = self.create_publisher(
            Wrench,
            '/haptic_device/force_feedback',
            10
        )

        # Timer for processing
        self.timer = self.create_timer(0.05, self.process_callback)  # 20Hz

        # State storage
        self.current_joints = None
        self.current_force = None
        self.current_pose = None
        self.simulated_environment = None
        self.contact_points = []

    def joint_state_callback(self, msg):
        """Process joint state from real manipulator"""
        self.current_joints = {
            'positions': msg.position,
            'velocities': msg.velocity,
            'efforts': msg.effort,
            'names': msg.name
        }

    def force_torque_callback(self, msg):
        """Process force/torque sensor data"""
        self.current_force = {
            'force': [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z],
            'torque': [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z],
            'timestamp': msg.header.stamp
        }

    def end_effector_pose_callback(self, msg):
        """Process end effector pose"""
        self.current_pose = {
            'position': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            'orientation': [msg.pose.orientation.x, msg.pose.orientation.y,
                           msg.pose.orientation.z, msg.pose.orientation.w]
        }

    def process_callback(self):
        """Process digital twin operations"""
        if self.current_joints is not None:
            # Update simulated manipulator to match real manipulator
            self.update_simulation()

        if self.current_force is not None and self.current_pose is not None:
            # Calculate haptic feedback based on environment interaction
            self.calculate_haptic_feedback()

    def update_simulation(self):
        """Send real manipulator state to simulation"""
        cmd_msg = Float64MultiArray()
        cmd_msg.data = list(self.current_joints['positions'])
        self.sim_joint_cmd_pub.publish(cmd_msg)

    def calculate_haptic_feedback(self):
        """Calculate haptic feedback based on force sensing and environment"""
        # Simulate contact forces based on end-effector position and environment
        if self.simulated_environment:
            # Calculate forces based on proximity to objects in simulation
            feedback_force = self.calculate_interaction_forces()
        else:
            # Use real force sensor data for haptic feedback
            feedback_force = Wrench()
            feedback_force.force.x = self.current_force['force'][0] * 0.1  # Scale down
            feedback_force.force.y = self.current_force['force'][1] * 0.1
            feedback_force.force.z = self.current_force['force'][2] * 0.1
            feedback_force.torque.x = self.current_force['torque'][0] * 0.1
            feedback_force.torque.y = self.current_force['torque'][1] * 0.1
            feedback_force.torque.z = self.current_force['torque'][2] * 0.1

        self.haptic_feedback_pub.publish(feedback_force)

    def calculate_interaction_forces(self):
        """Calculate interaction forces based on simulated environment"""
        # This would typically involve collision detection with simulated objects
        # For this example, we'll simulate contact with a virtual surface
        ee_pos = self.current_pose['position']

        # Virtual surface at z = 0.1m
        surface_z = 0.1
        distance_to_surface = max(0, ee_pos[2] - surface_z)

        # Generate repulsive force when close to surface
        if distance_to_surface < 0.05:  # Within 5cm
            force_magnitude = max(0, 0.05 - distance_to_surface) * 100  # Stiffness
            feedback_force = Wrench()
            feedback_force.force.z = -force_magnitude  # Push away from surface
            return feedback_force
        else:
            return Wrench()  # No force

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationDigitalTwin()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down manipulation digital twin...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Examples

### Smart Factory Assembly Line

Students implement a digital twin for an assembly line with multiple robotic arms performing coordinated tasks.

**Objectives:**
- Design digital twin architecture for multi-robot coordination
- Implement real-time synchronization between real and simulated robots
- Create visualization dashboard for monitoring assembly line status
- Develop predictive maintenance algorithms using digital twin data

**Required Components:**
- Industrial robot arms (real or simulated)
- Conveyor belt system
- Vision systems for quality control
- Assembly line components and fixtures
- Network infrastructure for real-time communication

**Evaluation Criteria:**
- Accuracy of digital twin synchronization
- Coordination effectiveness of multiple robots
- Response time to system changes
- Quality of visualization and monitoring interfaces

### Warehouse Logistics Digital Twin

Students create a digital twin for an automated warehouse with AMRs, conveyor systems, and inventory management.

**Objectives:**
- Model complex warehouse logistics in simulation
- Implement fleet management algorithms
- Integrate with real warehouse management systems
- Validate performance metrics against real data

**Required Components:**
- Autonomous mobile robots (AMRs)
- Warehouse management software
- Inventory tracking systems
- RFID or barcode scanning systems
- Communication infrastructure

**Evaluation Criteria:**
- Fleet utilization efficiency
- Order fulfillment accuracy
- System scalability under load
- Integration with existing warehouse systems

### Autonomous Delivery Robot Twin

Students develop a digital twin for an outdoor delivery robot with complex navigation and obstacle avoidance.

**Objectives:**
- Create realistic outdoor environment simulation
- Implement multi-sensor fusion for navigation
- Test safety protocols in simulation before real deployment
- Analyze performance in various weather conditions

**Required Components:**
- Delivery robot platform
- GPS and IMU sensors
- LIDAR and camera systems
- Weather simulation tools
- Safety validation protocols

**Evaluation Criteria:**
- Navigation accuracy and safety
- Adaptation to environmental changes
- Delivery success rate
- Energy efficiency optimization

## Summary

Chapter 9 provided comprehensive examples of digital twin implementations in robotics, covering industrial automation, fleet management, and manipulation tasks. Students learned to create synchronized systems that bridge real and simulated environments, implementing real-time data exchange and visualization. The practical examples demonstrated the value of digital twins in improving robot development, testing, and deployment across various applications.

## Quiz

1. What is a primary benefit of using digital twins in robotics?
   - A: Reduced hardware costs
   - B: Ability to test and validate in safe simulated environment
   - C: Faster robot movement
   - D: Simplified robot programming

   **Answer: B** - Digital twins allow for testing and validation in safe simulated environments before deploying to real robots, reducing risks and development time.

2. What does hardware-in-the-loop testing involve?
   - A: Testing without any hardware
   - B: Integrating real hardware with simulated environments
   - C: Testing only in simulation
   - D: Hardware that loops back on itself

   **Answer: B** - Hardware-in-the-loop testing involves integrating real hardware components with simulated environments to validate system behavior.

3. What is an important consideration when synchronizing real and simulated robots?
   - A: Robot color matching
   - B: Timing and data consistency
   - C: Same manufacturer requirements
   - D: Identical robot sizes

   **Answer: B** - Timing and data consistency are crucial when synchronizing real and simulated robots to ensure accurate representation.

4. Which metric would be most important for evaluating a fleet digital twin?
   - A: Individual robot weight
   - B: Fleet utilization efficiency
   - C: Battery brand used
   - D: Number of robot joints

   **Answer: B** - Fleet utilization efficiency is a key metric for evaluating fleet digital twins as it measures how effectively the robot fleet operates.

5. What is the purpose of comparing real vs. simulated performance?
   - A: To make simulation more complex
   - B: To validate simulation accuracy and identify discrepancies
   - C: To eliminate the need for real robots
   - D: To slow down the system

   **Answer: B** - Comparing real vs. simulated performance validates simulation accuracy and helps identify discrepancies that need to be addressed.

## Learning Outcomes

After completing this chapter, students will be able to:
- Create simulation environments for robot testing
- Implement physics-based simulations
- Bridge simulation and reality
- Validate robot behaviors in simulation

## Prerequisites

- Basic understanding of Python programming
- Fundamentals of linear algebra and calculus
- Basic knowledge of robotics concepts
- Introduction to machine learning concepts
- Completion of Module 0 (Introduction and Foundations)
- Completion of Chapter 01 (Physical AI Basics)
- Completion of Chapter 06 (Introduction to Digital Twins)
- Completion of Chapter 07 (Gazebo Simulation Basics)
- Completion of Chapter 08 (Integrating Unity for Visualization)

## Estimated Duration

6 hours