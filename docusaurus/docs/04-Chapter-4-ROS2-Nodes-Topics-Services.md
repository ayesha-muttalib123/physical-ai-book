---
id: 04-Chapter-4-ROS2-Nodes-Topics-Services
title: "Chapter 4: ROS2 Nodes, Topics, and Services"
sidebar_position: 4
---

# Chapter 4: ROS2 Nodes, Topics, and Services

## Overview

This chapter delves into the fundamental communication mechanisms in ROS2, focusing on nodes, topics, and services. Students will learn how to create ROS2 nodes, implement publisher-subscriber communication patterns, and design service-based interactions. The chapter covers the ROS2 architecture principles, lifecycle management, and best practices for designing robust communication systems in robotic applications.

## Why It Matters

Understanding ROS2 communication primitives is essential for building distributed robotic systems. Nodes, topics, and services form the backbone of ROS2 architecture, enabling modular, scalable, and maintainable robot software. Mastering these concepts allows developers to create complex systems where different components can communicate efficiently and reliably, forming the 'nervous system' of the robot.

## Key Concepts

### ROS2 Nodes
Independent processes that perform computation and communicate with other nodes. A node is the fundamental building block of a ROS2 system, containing the business logic for a specific function. Each node runs in its own process and can communicate with other nodes through topics, services, actions, and parameters.

### Topics
Asynchronous, many-to-many communication channels using publisher-subscriber pattern. Topics allow nodes to send and receive data in a decoupled manner. Publishers send messages to a topic, and any number of subscribers can receive those messages without knowing about each other.

### Services
Synchronous, request-response communication for direct interaction between nodes. Services provide a way for nodes to request specific actions or information from other nodes, similar to a traditional client-server model.

### ROS2 Client Libraries (RCL)
Language-specific libraries that provide ROS2 functionality. These libraries (like rclpy for Python and rclcpp for C++) provide the interface between your application code and the ROS2 middleware.

### DDS (Data Distribution Service)
Middleware that enables communication between ROS2 nodes. DDS is the underlying communication standard that ROS2 uses to manage data distribution between nodes, providing features like discovery, reliability, and quality of service.

### Node Lifecycle
The states and transitions that govern node behavior. ROS2 nodes can transition through various states (unconfigured, inactive, active, finalized) allowing for more sophisticated management of resources and behavior.

### Quality of Service (QoS) Profiles
Configuration parameters that define communication behavior. QoS profiles allow you to specify how messages should be handled in terms of reliability, durability, history, and other characteristics.

### Names and Namespaces
Hierarchical naming system for ROS2 entities. Namespaces provide a way to organize ROS2 entities and avoid naming conflicts, similar to directories in a file system.

## Code Examples

### ROS2 Publisher Node

Node that publishes sensor data to a topic:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publisher for sensor data
        self.publisher = self.create_publisher(LaserScan, 'sensor_scan', 10)

        # Create publisher for status
        self.status_publisher = self.create_publisher(String, 'status', 10)

        # Create timer to publish data at 10Hz
        self.timer = self.create_timer(0.1, self.publish_scan_data)

        self.get_logger().info('Sensor publisher node initialized')

    def publish_scan_data(self):
        # Create and populate LaserScan message
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'

        # Set scan parameters
        scan_msg.angle_min = -1.57  # -90 degrees
        scan_msg.angle_max = 1.57   # 90 degrees
        scan_msg.angle_increment = 0.0174  # 1 degree

        # Generate sample ranges (simulated sensor data)
        num_ranges = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment)
        scan_msg.ranges = [random.uniform(0.1, 10.0) for _ in range(num_ranges)]

        # Publish the scan data
        self.publisher.publish(scan_msg)

        # Publish status message
        status_msg = String()
        status_msg.data = f'Published scan with {len(scan_msg.ranges)} ranges'
        self.status_publisher.publish(status_msg)

        self.get_logger().info(f'Published scan data: {status_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    sensor_publisher = SensorPublisher()

    try:
        rclpy.spin(sensor_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_publisher.destroy_node()
        rclpy.shutdown()
```

### ROS2 Subscriber Node

Node that subscribes to sensor data and processes it:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')

        # Create subscription to sensor data
        self.subscription = self.create_subscription(
            LaserScan,
            'sensor_scan',
            self.scan_callback,
            10
        )

        # Create publisher for processed data
        self.min_distance_publisher = self.create_publisher(Float32, 'min_distance', 10)
        self.obstacle_warning_publisher = self.create_publisher(String, 'obstacle_warning', 10)

        self.get_logger().info('Sensor subscriber node initialized')

    def scan_callback(self, msg):
        # Process the scan data
        if len(msg.ranges) > 0:
            # Filter out invalid ranges (inf, nan)
            valid_ranges = [r for r in msg.ranges if not (r == float('inf') or r == float('nan'))]

            if valid_ranges:
                min_distance = min(valid_ranges)
                max_distance = max(valid_ranges)

                # Publish minimum distance
                min_dist_msg = Float32()
                min_dist_msg.data = min_distance
                self.min_distance_publisher.publish(min_dist_msg)

                # Check for obstacles
                if min_distance < 1.0:  # Obstacle within 1 meter
                    from std_msgs.msg import String
                    warning_msg = String()
                    warning_msg.data = f'OBSTACLE DETECTED: {min_distance:.2f}m'
                    self.obstacle_warning_publisher.publish(warning_msg)
                    self.get_logger().warn(f'Obstacle detected: {min_distance:.2f}m')
                else:
                    self.get_logger().info(f'Min distance: {min_distance:.2f}m, Max distance: {max_distance:.2f}m')
            else:
                self.get_logger().warn('No valid ranges in scan data')

def main(args=None):
    rclpy.init(args=args)
    sensor_subscriber = SensorSubscriber()

    try:
        rclpy.spin(sensor_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_subscriber.destroy_node()
        rclpy.shutdown()
```

### ROS2 Service Server and Client

Service for controlling robot movement with request-response pattern:

```python
# Service definition file (srv/MovementControl.srv):
# float64 linear_velocity
# float64 angular_velocity
# ---
# bool success
# string message

# Service Server
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger  # Using Trigger as simple example
from std_msgs.msg import String

class MovementControlService(Node):
    def __init__(self):
        super().__init__('movement_control_service')

        # Create service
        self.srv = self.create_service(
            Trigger,  # In practice, use custom service type
            'move_robot',
            self.move_robot_callback
        )

        # Publisher for movement commands
        self.cmd_publisher = self.create_publisher(String, 'robot_commands', 10)

        self.get_logger().info('Movement control service ready')

    def move_robot_callback(self, request, response):
        # In a real implementation, this would process movement commands
        # For this example, we'll just simulate movement
        self.get_logger().info('Received movement request')

        # Publish movement command
        cmd_msg = String()
        cmd_msg.data = 'move_forward_1m'
        self.cmd_publisher.publish(cmd_msg)

        # Set response
        response.success = True
        response.message = 'Movement command executed successfully'

        return response

# Service Client
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
import time

class MovementClient(Node):
    def __init__(self):
        super().__init__('movement_client')
        self.cli = self.create_client(Trigger, 'move_robot')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main_server(args=None):
    rclpy.init(args=args)
    service = MovementControlService()

    try:
        rclpy.spin(service)
    except KeyboardInterrupt:
        pass
    finally:
        service.destroy_node()
        rclpy.shutdown()

def main_client(args=None):
    rclpy.init(args=args)
    client = MovementClient()

    response = client.send_request()
    if response:
        print(f'Response: {response.success}, {response.message}')
    else:
        print('Service call failed')

    client.destroy_node()
    rclpy.shutdown()
```

## Practical Examples

### Multi-Robot Communication System

Students implement a system where multiple robots communicate using ROS2 topics and services to coordinate their movements and share sensor data.

**Objectives:**
- Design ROS2 nodes for multi-robot communication
- Implement publisher-subscriber patterns for sensor sharing
- Create services for coordination requests

**Required Components:**
- Multiple simulated robots
- ROS2 network configuration
- Communication protocols

**Evaluation Criteria:**
- Successful inter-robot communication
- Efficient data sharing
- Coordination effectiveness

### Sensor Fusion Node

Students create a ROS2 node that subscribes to multiple sensor topics and publishes fused sensor data.

**Objectives:**
- Implement multiple subscribers in one node
- Synchronize data from different sensors
- Publish processed/fused data

**Required Components:**
- Multiple sensor topics
- Synchronization algorithms
- Fusion algorithms

**Evaluation Criteria:**
- Correct data synchronization
- Effective sensor fusion
- Robust node implementation

### Robot Control Service

Students develop a service-based system for remote robot control with safety checks.

**Objectives:**
- Implement a ROS2 service for robot control
- Add safety validation to service responses
- Create client nodes to call the service

**Required Components:**
- Robot simulation
- Safety validation logic
- Client-server communication

**Evaluation Criteria:**
- Secure service implementation
- Effective safety checks
- Reliable communication

## Summary

Chapter 3 covers the fundamental ROS2 communication mechanisms: nodes, topics, and services. Students learned how to create ROS2 nodes, implement publisher-subscriber patterns, and design service-based interactions. Through practical examples, they gained experience with multi-robot communication, sensor fusion, and service-based control systems. These concepts form the backbone of ROS2 architecture and are essential for building distributed robotic systems.

## Quiz

1. What is the main difference between ROS2 topics and services?
   - A: Topics are for hardware, services are for software
   - B: Topics are asynchronous many-to-many, services are synchronous request-response
   - C: Topics are faster than services
   - D: There is no difference between topics and services

   **Answer: B** - Topics use an asynchronous, many-to-many communication pattern (publisher-subscriber), while services use a synchronous, request-response pattern between client and server.

2. What is a ROS2 node?
   - A: A type of robot hardware
   - B: An independent process that performs computation and communicates with other nodes
   - C: A configuration file for ROS2
   - D: A special type of message

   **Answer: B** - A ROS2 node is an independent process that performs computation and communicates with other nodes through topics, services, actions, and parameters.

3. What does DDS stand for in the context of ROS2?
   - A: Distributed Data System
   - B: Data Distribution Service
   - C: Dynamic Discovery System
   - D: Distributed Deployment Service

   **Answer: B** - DDS stands for Data Distribution Service, which is the middleware that enables communication between ROS2 nodes.

4. What is Quality of Service (QoS) in ROS2?
   - A: A measure of code quality
   - B: Configuration parameters that define communication behavior
   - C: A debugging tool
   - D: A type of ROS2 message

   **Answer: B** - Quality of Service (QoS) profiles are configuration parameters that define communication behavior such as reliability, durability, and history for topics and services.

5. What is the purpose of namespaces in ROS2?
   - A: To organize files on disk
   - B: To provide hierarchical naming for ROS2 entities to avoid naming conflicts
   - C: To improve performance
   - D: To encrypt messages

   **Answer: B** - Namespaces provide a hierarchical naming system for ROS2 entities to organize them and avoid naming conflicts between different nodes and topics.

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

## Estimated Duration

6 hours