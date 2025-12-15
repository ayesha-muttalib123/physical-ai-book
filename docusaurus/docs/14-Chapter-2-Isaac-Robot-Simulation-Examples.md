---
id: 14-Chapter-2-Isaac-Robot-Simulation-Examples
title: "Chapter 2: Isaac Robot Simulation Examples"
sidebar_position: 14
---

# Chapter 2: Isaac Robot Simulation Examples

## Overview

This chapter provides comprehensive examples of robot simulation using NVIDIA Isaac Sim, demonstrating how to create realistic robotic scenarios with high-fidelity physics, sensor simulation, and AI training environments. Students will learn to build complex simulation scenes, configure robot models with accurate physics properties, and implement sensor simulation pipelines. The chapter covers both basic and advanced simulation techniques, including synthetic data generation for AI model training and hardware-in-the-loop testing.

## Why It Matters

Simulation is crucial for robotics development as it allows for rapid prototyping, testing, and training without the risks and costs associated with real hardware. Isaac Sim provides industry-leading physics simulation, photorealistic rendering, and GPU-accelerated sensor simulation that enables the creation of highly realistic virtual environments. These capabilities are essential for developing robust robotics systems and training AI models with synthetic data.

## Key Concepts

### Isaac Sim Environment Creation
Building realistic simulation environments. This involves creating detailed 3D environments with accurate physics properties, lighting, and materials that closely match real-world conditions.

### Physics-Based Simulation
Accurate physics modeling for robot interactions. Isaac Sim uses advanced physics engines to simulate real-world forces, collisions, and material properties, enabling realistic robot behavior and interaction with the environment.

### Sensor Simulation
GPU-accelerated camera, LIDAR, and IMU simulation. Isaac Sim provides realistic sensor models that simulate the behavior of real sensors, including noise, distortion, and other physical effects.

### Synthetic Data Generation
Creating labeled training data from simulation. This involves using simulation environments to generate large amounts of training data with perfect ground truth labels, which is essential for training AI models.

### Hardware-in-the-Loop
Connecting real hardware to simulated environments. This technique allows developers to test real hardware components with simulated environments, enabling safer and more cost-effective testing.

### Multi-Robot Simulation
Simulating multiple robots in shared environments. This allows for testing of coordination, communication, and fleet management systems in a controlled environment.

### Dynamic Scene Elements
Simulating moving objects and changing environments. This includes creating environments with dynamic elements like moving obstacles, changing lighting conditions, and evolving scenarios.

### Performance Optimization
Techniques for efficient simulation execution. This involves optimizing simulation parameters, using appropriate level-of-detail models, and managing computational resources effectively.

## Code Examples

### Isaac Sim Robot Setup and Control

Complete example of setting up a robot in Isaac Sim with physics and sensor simulation:

```python
#!/usr/bin/env python3
"""
Isaac Sim Robot Setup and Control Example
Demonstrates how to set up a robot in Isaac Sim with physics and sensors
"""
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrim, Articulation
import numpy as np
import carb
import asyncio
import omni.replicator.core as rep

class IsaacSimRobotExample:
    def __init__(self):
        self.world = None
        self.robot = None
        self.camera = None
        self.objects = []
        self.simulation_steps = 0

    def setup_world(self):
        """Initialize the Isaac Sim world with robot and environment"""
        # Create world with 60Hz physics update rate
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

        # Add default ground plane
        self.world.scene.add_default_ground_plane()

        # Set up the viewport camera view
        set_camera_view(eye=np.array([2.5, 2.5, 2.0]), target=np.array([0, 0, 0.5]))

        # Add a simple environment with obstacles
        self.setup_environment()

        # Add robot (using a simple differential drive robot as example)
        self.setup_robot()

        # Add sensors to robot
        self.setup_sensors()

        carb.log_info("Isaac Sim world setup complete")

    def setup_environment(self):
        """Setup the simulation environment with objects"""
        # Add static obstacles
        obstacle1 = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Obstacle1",
                name="obstacle1",
                position=np.array([1.0, 0.5, 0.25]),
                size=0.5,
                color=np.array([0.5, 0.5, 0.5])
            )
        )

        obstacle2 = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Obstacle2",
                name="obstacle2",
                position=np.array([-0.8, -0.5, 0.25]),
                size=0.4,
                color=np.array([0.7, 0.3, 0.3])
            )
        )

        # Add physics material
        material = PhysicsMaterial(
            prim_path="/World/physics_material",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1
        )

        obstacle1.set_material(material)
        obstacle2.set_material(material)

        # Add some dynamic objects for interaction
        for i in range(3):
            dynamic_obj = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/DynamicObj{i}",
                    name=f"dynamic_obj_{i}",
                    position=np.array([0.5 + i*0.3, 1.0, 0.5]),
                    size=0.15,
                    color=np.array([0.2, 0.6, 0.8])
                )
            )
            dynamic_obj.set_material(material)
            self.objects.append(dynamic_obj)

    def setup_robot(self):
        """Setup the robot in the simulation"""
        # For this example, we'll create a simple differential drive robot
        # In real applications, you would load a URDF or USD robot model

        # Create robot body
        robot_body = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/RobotBody",
                name="robot_body",
                position=np.array([0.0, 0.0, 0.3]),
                size=np.array([0.3, 0.4, 0.2]),
                color=np.array([0.1, 0.1, 0.8])
            )
        )

        # Add wheels (simplified as cuboids)
        # Left wheel
        left_wheel = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/RobotBody/LeftWheel",
                name="left_wheel",
                position=np.array([-0.15, -0.25, 0.1]),
                size=np.array([0.1, 0.1, 0.2]),
                color=np.array([0.3, 0.3, 0.3])
            )
        )

        # Right wheel
        right_wheel = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/RobotBody/RightWheel",
                name="right_wheel",
                position=np.array([-0.15, 0.25, 0.1]),
                size=np.array([0.1, 0.1, 0.2]),
                color=np.array([0.3, 0.3, 0.3])
            )
        )

        # Store robot reference
        self.robot = robot_body

        carb.log_info("Robot setup complete")

    def setup_sensors(self):
        """Setup sensors on the robot"""
        # Add RGB camera
        self.camera = Camera(
            prim_path="/World/RobotBody/Camera",
            name="robot_camera",
            position=np.array([0.1, 0.0, 0.15]),
            frequency=20,  # 20Hz
            resolution=(640, 480)
        )
        self.camera.initialize()
        self.camera.add_render_product("/World/RobotBody/Camera", [640, 480])

        carb.log_info("Sensors setup complete")

    def control_robot(self, linear_vel, angular_vel):
        """Simple robot control (differential drive approximation)"""
        if self.robot is None:
            return

        # Get current position and orientation
        current_pos, current_ori = self.robot.get_world_pose()

        # Calculate movement based on velocities
        dt = 1.0/60.0  # Physics timestep

        # Simple kinematic model for differential drive
        # Move forward/backward
        new_x = current_pos[0] + linear_vel * np.cos(current_ori[2]) * dt
        new_y = current_pos[1] + linear_vel * np.sin(current_ori[2]) * dt

        # Rotate
        new_theta = current_ori[2] + angular_vel * dt

        # Update robot position
        self.robot.set_world_pose(
            position=np.array([new_x, new_y, current_pos[2]]),
            orientation=np.array([0, 0, np.sin(new_theta/2), np.cos(new_theta/2)])
        )

    def run_simulation(self, steps=1000):
        """Run the simulation for specified number of steps"""
        carb.log_info(f"Starting simulation for {steps} steps...")

        # Reset the world
        self.world.reset()

        # Run simulation loop
        for step in range(steps):
            # Simple control logic - move in square pattern
            cycle = (step // 300) % 4  # Change direction every 300 steps

            if cycle == 0:  # Move forward
                linear_vel = 0.5
                angular_vel = 0.0
            elif cycle == 1:  # Turn right
                linear_vel = 0.0
                angular_vel = -0.5
            elif cycle == 2:  # Move forward
                linear_vel = 0.5
                angular_vel = 0.0
            else:  # Turn right
                linear_vel = 0.0
                angular_vel = -0.5

            # Apply control
            self.control_robot(linear_vel, angular_vel)

            # Step the world
            self.world.step(render=True)

            # Process camera data every 10 steps
            if step % 10 == 0:
                self.process_camera_data()

            self.simulation_steps += 1

        carb.log_info(f"Simulation completed after {steps} steps")

    def process_camera_data(self):
        """Process camera data from simulation"""
        try:
            # Get camera data
            camera_data = self.camera.get_rgb()
            if camera_data is not None:
                height, width, channels = camera_data.shape
                carb.log_info(f"Camera data: {width}x{height}x{channels}")

                # In real applications, you would process this data for:
                # - Object detection
                # - SLAM algorithms
                # - Training data generation
                # - Visualization

        except Exception as e:
            carb.log_error(f"Error getting camera data: {e}")

    def cleanup(self):
        """Clean up simulation resources"""
        if self.world:
            self.world.clear()
        carb.log_info("Simulation cleanup complete")

def main():
    """Main function to run the Isaac Sim robot example"""
    carb.log_info("Starting Isaac Sim Robot Example...")

    # Create simulation example
    sim_example = IsaacSimRobotExample()

    try:
        # Setup the world
        sim_example.setup_world()

        # Run simulation
        sim_example.run_simulation(steps=1200)  # Run for 20 seconds at 60Hz

        # Cleanup
        sim_example.cleanup()

    except Exception as e:
        carb.log_error(f"Error in simulation: {e}")
    finally:
        carb.log_info("Isaac Sim Robot Example completed")

if __name__ == "__main__":
    main()
```

### Advanced Isaac Sim with Replicator

Advanced example using Isaac Replicator for synthetic data generation:

```python
#!/usr/bin/env python3
"""
Advanced Isaac Sim with Replicator Example
Demonstrates synthetic data generation for AI training
"""
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
import numpy as np
import carb
import omni.replicator.core as rep
import omni.kit.commands
from PIL import Image
import os
import json
from pxr import Gf, Sdf, UsdGeom

class IsaacReplicatorExample:
    def __init__(self, output_dir="./synthetic_data"):
        self.world = None
        self.output_dir = output_dir
        self.camera = None
        self.light = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def setup_world(self):
        """Initialize the Isaac Sim world for data generation"""
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/30.0)

        # Add default ground plane
        self.world.scene.add_default_ground_plane()

        # Set up the viewport camera view
        set_camera_view(eye=np.array([3.0, 3.0, 2.0]), target=np.array([0, 0, 0.5]))

        # Add dynamic lighting
        self.setup_lighting()

        # Create diverse scene with multiple objects
        self.setup_diverse_scene()

        # Setup replicator
        self.setup_replicator()

        carb.log_info("Advanced Isaac Sim world setup complete")

    def setup_lighting(self):
        """Setup dynamic lighting for diverse data generation"""
        # Add dome light
        with rep.new_layer():
            # Create dome light with random properties
            dome_light = rep.create.light(
                light_type="Dome",
                color=rep.distribution.uniform((0.2, 0.2, 0.2), (1.0, 1.0, 1.0)),
                intensity=rep.distribution.normal(3000, 500),
                texture_begin=rep.distribution.uniform(0, 360),
                texture_end=rep.distribution.uniform(0, 360)
            )

            with dome_light:
                rep.modify.visibility(rep.distribution.choice([True, False], [0.8, 0.2]))

            # Add distant light
            distant_light = rep.create.light(
                light_type="Distant",
                color=rep.distribution.uniform((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
                intensity=rep.distribution.normal(4000, 1000)
            )
            with distant_light:
                rep.modify.pose(
                    position=rep.distribution.uniform((-5, -5, 5), (5, 5, 10)),
                    look_at=(0, 0, 0)
                )

    def setup_diverse_scene(self):
        """Setup a diverse scene with various objects for data generation"""
        # Create object templates
        object_configs = [
            {"size": 0.2, "color": (0.8, 0.2, 0.2), "shape": "cube", "position": (-1.0, -0.5, 0.2)},
            {"size": 0.15, "color": (0.2, 0.8, 0.2), "shape": "cube", "position": (0.5, 0.8, 0.15)},
            {"size": 0.25, "color": (0.2, 0.2, 0.8), "shape": "cube", "position": (1.2, -0.3, 0.25)},
            {"size": 0.18, "color": (0.8, 0.8, 0.2), "shape": "cube", "position": (-0.2, 1.0, 0.18)},
            {"size": 0.22, "color": (0.8, 0.2, 0.8), "shape": "cube", "position": (0.8, 0.5, 0.22)},
        ]

        # Add objects to scene
        for i, config in enumerate(object_configs):
            if config["shape"] == "cube":
                obj = self.world.scene.add(
                    DynamicCuboid(
                        prim_path=f"/World/Object{i}",
                        name=f"object_{i}",
                        position=np.array(config["position"]),
                        size=config["size"],
                        color=np.array(config["color"])
                    )
                )

        # Add physics material
        material = PhysicsMaterial(
            prim_path="/World/physics_material",
            static_friction=rep.distribution.uniform(0.1, 0.9),
            dynamic_friction=rep.distribution.uniform(0.1, 0.9),
            restitution=rep.distribution.uniform(0.0, 0.5)
        )

    def setup_replicator(self):
        """Setup Isaac Replicator for synthetic data generation"""
        # Enable replicator
        rep.orchestrator._orchestrator = None

        # Create camera for data generation
        camera = rep.create.camera(
            position=rep.distribution.uniform((-2, -2, 1), (2, 2, 3)),
            look_at=rep.distribution.uniform((-0.5, -0.5, 0), (0.5, 0.5, 1))
        )

        # Create render product
        render_product = rep.create.render_product(
            camera,
            (1024, 1024),
            name="synthetic_data_camera"
        )

        # Add various sensors for data generation
        with rep.new_layer():
            # RGB data
            rep.WriterRegistry.enable_writer("Rgb", device="cpu", output_dir=self.output_dir + "/rgb")

            # Semantic segmentation
            rep.WriterRegistry.enable_writer("SemanticSegmentation", device="cpu", output_dir=self.output_dir + "/segmentation")

            # Bounding box 2D
            rep.WriterRegistry.enable_writer("BoundingBox2D", device="cpu", output_dir=self.output_dir + "/bbox_2d")

            # Depth data
            rep.WriterRegistry.enable_writer("DistanceToImagePlane", device="cpu", output_dir=self.output_dir + "/depth")

    def setup_annotators(self):
        """Setup annotators for different types of synthetic data"""
        # Enable various annotators
        rep.orchestrator.add_sensors([rep.Sensors.CAMERA, rep.Sensors.RAYCAM])

        # Setup semantic segmentation
        rep.orchestrator.add_annotators([
            rep.Annotators.SEMANTIC_SEGMENTATION,
            rep.Annotators.BOUNDING_BOX_2D,
            rep.Annotators.DEPTH,
            rep.Annotators.RGB
        ])

    def generate_synthetic_data(self, num_samples=100):
        """Generate synthetic data using Isaac Replicator"""
        carb.log_info(f"Generating {num_samples} synthetic data samples...")

        # Reset the world
        self.world.reset()

        # Initialize replicator
        rep.orchestrator.setup_camera("/World/Camera")
        self.setup_annotators()

        # Generate data
        with rep.trigger.on_frame(num_frames=num_samples):
            # Randomize camera positions
            with rep.get.camera(path=".*"):
                rep.modify.pose(
                    position=rep.distribution.uniform((-2, -2, 1), (2, 2, 3)),
                    look_at=rep.distribution.uniform((-0.5, -0.5, 0), (0.5, 0.5, 1))
                )

            # Randomize lighting
            with rep.get.light(path=".*"):
                rep.modify.light(
                    color=rep.distribution.uniform((0.3, 0.3, 0.3), (1.0, 1.0, 1.0)),
                    intensity=rep.distribution.normal(3000, 500)
                )

            # Randomize object positions slightly
            for i in range(5):  # For each object
                with rep.get.prims(path=f"/World/Object{i}"):
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (-1.5, -1.5, 0.1), (1.5, 1.5, 0.5)
                        )
                    )

        # Run the replicator
        rep.orchestrator.run()
        carb.log_info(f"Synthetic data generation completed. Data saved to {self.output_dir}")

    def run_advanced_simulation(self):
        """Run the advanced simulation with data generation"""
        try:
            # Setup the world
            self.setup_world()

            # Generate synthetic data
            self.generate_synthetic_data(num_samples=50)

            # Additional simulation for dynamics
            carb.log_info("Running additional dynamics simulation...")
            for i in range(300):  # 5 seconds at 60Hz
                self.world.step(render=True)
                if i % 60 == 0:  # Log every second
                    carb.log_info(f"Simulation step {i}/300")

        except Exception as e:
            carb.log_error(f"Error in advanced simulation: {e}")
        finally:
            if self.world:
                self.world.clear()
            carb.log_info("Advanced Isaac Sim example completed")

def main():
    """Main function to run the advanced Isaac Sim example"""
    carb.log_info("Starting Advanced Isaac Sim with Replicator Example...")

    # Create advanced simulation example
    advanced_sim = IsaacReplicatorExample(output_dir="./advanced_synthetic_data")

    # Run the example
    advanced_sim.run_advanced_simulation()

if __name__ == "__main__":
    main()
```

### Multi-Robot Isaac Sim Environment

Example of simulating multiple robots in a shared Isaac Sim environment:

```python
#!/usr/bin/env python3
"""
Multi-Robot Isaac Sim Environment
Demonstrates simulating multiple robots in a shared environment
"""
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.sensors import Camera
import numpy as np
import carb
import asyncio
from typing import List, Dict
import time

class MultiRobotSimulator:
    def __init__(self, num_robots=3):
        self.world = None
        self.robots = {}
        self.cameras = {}
        self.environment_objects = []
        self.num_robots = num_robots
        self.simulation_steps = 0

        # Robot goals for navigation
        self.robot_goals = {}
        self.robot_states = {}

    def setup_world(self):
        """Initialize the multi-robot simulation world"""
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

        # Add default ground plane
        self.world.scene.add_default_ground_plane()

        # Set up the viewport camera view
        set_camera_view(eye=np.array([4.0, 4.0, 3.0]), target=np.array([0, 0, 0.5]))

        # Create environment
        self.setup_environment()

        # Create multiple robots
        self.setup_robots()

        # Setup robot-specific sensors
        self.setup_robot_sensors()

        carb.log_info(f"Multi-robot simulation setup complete with {self.num_robots} robots")

    def setup_environment(self):
        """Setup the shared environment with obstacles"""
        # Create a warehouse-like environment
        obstacles_config = [
            {"position": [2.0, 0.0, 0.5], "size": [0.5, 3.0, 1.0], "color": [0.5, 0.5, 0.5]},
            {"position": [-2.0, 0.0, 0.5], "size": [0.5, 3.0, 1.0], "color": [0.5, 0.5, 0.5]},
            {"position": [0.0, 2.0, 0.5], "size": [3.0, 0.5, 1.0], "color": [0.5, 0.5, 0.5]},
            {"position": [0.0, -2.0, 0.5], "size": [3.0, 0.5, 1.0], "color": [0.5, 0.5, 0.5]},
            {"position": [1.0, 1.0, 0.25], "size": [0.5, 0.5, 0.5], "color": [0.7, 0.3, 0.3]},
            {"position": [-1.0, -1.0, 0.25], "size": [0.5, 0.5, 0.5], "color": [0.3, 0.7, 0.3]},
        ]

        for i, config in enumerate(obstacles_config):
            obstacle = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Obstacle{i}",
                    name=f"obstacle_{i}",
                    position=np.array(config["position"]),
                    size=np.array(config["size"]),
                    color=np.array(config["color"])
                )
            )
            self.environment_objects.append(obstacle)

        # Add physics material
        material = PhysicsMaterial(
            prim_path="/World/physics_material",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.1
        )

        for obj in self.environment_objects:
            obj.set_material(material)

    def setup_robots(self):
        """Setup multiple robots in the environment"""
        # Define starting positions for robots in a circle
        center = np.array([0.0, 0.0, 0.3])
        radius = 1.5

        for i in range(self.num_robots):
            angle = 2 * np.pi * i / self.num_robots
            start_pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

            # Create robot body
            robot = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Robot{i}",
                    name=f"robot_{i}",
                    position=start_pos,
                    size=np.array([0.3, 0.3, 0.2]),
                    color=np.array([0.1 + 0.3*i, 0.1 + 0.2*i, 0.8 - 0.1*i])  # Different colors
                )
            )

            # Store robot reference
            self.robots[f"robot_{i}"] = robot

            # Initialize robot state
            self.robot_states[f"robot_{i}"] = {
                "position": start_pos,
                "orientation": 0.0,
                "linear_vel": 0.0,
                "angular_vel": 0.0,
                "status": "idle"
            }

            # Set random goals for each robot
            goal_angle = 2 * np.pi * (i + 1) / self.num_robots
            goal_pos = center + np.array([radius * np.cos(goal_angle + np.pi),
                                          radius * np.sin(goal_angle + np.pi), 0])
            self.robot_goals[f"robot_{i}"] = goal_pos

    def setup_robot_sensors(self):
        """Setup sensors for each robot"""
        for i in range(self.num_robots):
            # Add camera to each robot
            camera = Camera(
                prim_path=f"/World/Robot{i}/Camera",
                name=f"robot_{i}_camera",
                position=np.array([0.15, 0.0, 0.1]),
                frequency=10,  # 10Hz
                resolution=(320, 240)
            )
            camera.initialize()
            camera.add_render_product(f"/World/Robot{i}/Camera", [320, 240])

            self.cameras[f"robot_{i}"] = camera

    def simple_navigation_controller(self, robot_id: str):
        """Simple navigation controller for a robot"""
        robot_state = self.robot_states[robot_id]
        goal_pos = self.robot_goals[robot_id]

        # Get current position
        current_pos, current_ori = self.robots[robot_id].get_world_pose()

        # Calculate direction to goal
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)

        # Simple proportional controller
        if distance > 0.2:  # If not close to goal
            # Calculate desired angle
            desired_angle = np.arctan2(dy, dx)

            # Get current orientation angle
            current_angle = np.arctan2(
                2 * (current_ori[3] * current_ori[2] + current_ori[0] * current_ori[1]),
                1 - 2 * (current_ori[1]**2 + current_ori[2]**2)
            )

            # Calculate angle difference
            angle_diff = desired_angle - current_angle
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # Set velocities
            linear_vel = min(distance * 0.5, 0.3)  # Max 0.3 m/s
            angular_vel = max(min(angle_diff * 1.0, 0.5), -0.5)  # Max 0.5 rad/s

            robot_state["linear_vel"] = linear_vel
            robot_state["angular_vel"] = angular_vel
            robot_state["status"] = "navigating"
        else:
            # Reached goal
            robot_state["linear_vel"] = 0.0
            robot_state["angular_vel"] = 0.0
            robot_state["status"] = "goal_reached"

            # Set a new random goal after reaching current one
            center = np.array([0.0, 0.0, 0.3])
            new_angle = np.random.uniform(0, 2*np.pi)
            new_radius = np.random.uniform(1.0, 2.0)
            new_goal = center + np.array([new_radius * np.cos(new_angle),
                                          new_radius * np.sin(new_angle), 0])
            self.robot_goals[robot_id] = new_goal
            robot_state["status"] = "setting_new_goal"

    def update_robot_motion(self):
        """Update motion for all robots"""
        dt = 1.0/60.0  # Physics timestep

        for robot_id, robot in self.robots.items():
            state = self.robot_states[robot_id]

            if state["status"] in ["navigating", "setting_new_goal"]:
                # Get current pose
                current_pos, current_ori = robot.get_world_pose()

                # Calculate new position based on velocities
                linear_vel = state["linear_vel"]
                angular_vel = state["angular_vel"]

                # Calculate new orientation
                current_angle = np.arctan2(
                    2 * (current_ori[3] * current_ori[2] + current_ori[0] * current_ori[1]),
                    1 - 2 * (current_ori[1]**2 + current_ori[2]**2)
                )
                new_angle = current_angle + angular_vel * dt

                # Calculate new position
                new_x = current_pos[0] + linear_vel * np.cos(new_angle) * dt
                new_y = current_pos[1] + linear_vel * np.sin(new_angle) * dt
                new_z = current_pos[2]  # Keep same height

                # Update robot pose
                robot.set_world_pose(
                    position=np.array([new_x, new_y, new_z]),
                    orientation=np.array([0, 0, np.sin(new_angle/2), np.cos(new_angle/2)])
                )

                # Update state
                state["position"] = np.array([new_x, new_y, new_z])
                state["orientation"] = new_angle

    def run_simulation(self, steps=3600):  # 1 minute at 60Hz
        """Run the multi-robot simulation"""
        carb.log_info(f"Starting multi-robot simulation for {steps} steps...")

        # Reset the world
        self.world.reset()

        # Run simulation loop
        for step in range(steps):
            # Update each robot's navigation
            for robot_id in self.robots.keys():
                self.simple_navigation_controller(robot_id)

            # Update robot motions
            self.update_robot_motion()

            # Step the world
            self.world.step(render=True)

            # Log status every 600 steps (10 seconds)
            if step % 600 == 0:
                self.log_simulation_status(step)

            self.simulation_steps += 1

        carb.log_info(f"Multi-robot simulation completed after {steps} steps")

    def log_simulation_status(self, step):
        """Log the current simulation status"""
        status_msg = f"Step {step}: "
        for robot_id, state in self.robot_states.items():
            goal = self.robot_goals[robot_id]
            pos = state["position"]
            distance = np.linalg.norm(goal[:2] - pos[:2])
            status_msg += f"{robot_id}({state['status']}, dist_to_goal:{distance:.2f}) "

        carb.log_info(status_msg)

    def cleanup(self):
        """Clean up simulation resources"""
        if self.world:
            self.world.clear()
        carb.log_info("Multi-robot simulation cleanup complete")

def main():
    """Main function to run the multi-robot Isaac Sim example"""
    carb.log_info("Starting Multi-Robot Isaac Sim Example...")

    # Create multi-robot simulation
    multi_sim = MultiRobotSimulator(num_robots=4)  # 4 robots

    try:
        # Setup the world
        multi_sim.setup_world()

        # Run simulation
        multi_sim.run_simulation(steps=3600)  # Run for 1 minute

        # Cleanup
        multi_sim.cleanup()

    except Exception as e:
        carb.log_error(f"Error in multi-robot simulation: {e}")
    finally:
        carb.log_info("Multi-Robot Isaac Sim Example completed")

if __name__ == "__main__":
    main()
```

## Practical Examples

### Warehouse Automation Simulation

Students create a comprehensive warehouse automation simulation with multiple robots, conveyor systems, and inventory management.

**Objectives:**
- Build realistic warehouse environment in Isaac Sim
- Implement multi-robot coordination and path planning
- Create conveyor belt and inventory tracking systems
- Optimize robot fleet performance

**Required Components:**
- Isaac Sim installation
- Warehouse asset models
- Mobile robot models
- Inventory tracking systems
- Fleet management algorithms

**Evaluation Criteria:**
- Environment realism and complexity
- Robot coordination effectiveness
- System performance optimization
- Fleet utilization efficiency

### Autonomous Vehicle Testing

Students develop an autonomous vehicle testing environment with complex traffic scenarios and sensor simulation.

**Objectives:**
- Create realistic urban driving environment
- Implement traffic simulation with other vehicles
- Simulate various sensor modalities (camera, LIDAR, radar)
- Test navigation and obstacle avoidance

**Required Components:**
- Vehicle dynamics models
- Urban environment assets
- Traffic simulation tools
- Multi-sensor simulation
- Path planning algorithms

**Evaluation Criteria:**
- Environment fidelity
- Sensor simulation accuracy
- Navigation performance
- Safety and reliability

### Robotic Manipulation Training

Students create a robotic manipulation training environment for AI model development using synthetic data.

**Objectives:**
- Build diverse manipulation scenes
- Generate synthetic training data with annotations
- Implement grasp planning and execution
- Validate performance on real hardware

**Required Components:**
- Manipulator robot models
- Object asset libraries
- Grasp planning algorithms
- Synthetic data generation tools
- Real robot for validation

**Evaluation Criteria:**
- Data diversity and quality
- Grasp success rate improvement
- Simulation-to-reality transfer
- Training effectiveness

## Summary

Chapter 13 provided comprehensive examples of robot simulation using NVIDIA Isaac Sim, covering robot setup and control, advanced replicator usage for synthetic data generation, and multi-robot coordination. Students learned to create realistic simulation environments with accurate physics, implement sensor simulation, and generate synthetic data for AI training. The practical examples demonstrated real-world applications of Isaac Sim in warehouse automation, autonomous vehicle testing, and robotic manipulation training.

## Quiz

1. What is a key advantage of Isaac Sim for robotics development?
   - A: Lower hardware costs only
   - B: High-fidelity physics simulation and photorealistic rendering
   - C: Simpler programming requirements
   - D: Reduced need for sensors

   **Answer: B** - Isaac Sim provides high-fidelity physics simulation and photorealistic rendering, essential for realistic robotics testing and training.

2. What does Isaac Replicator enable?
   - A: Hardware control only
   - B: Synthetic data generation for AI training
   - C: Robot movement only
   - D: Communication protocols

   **Answer: B** - Isaac Replicator enables synthetic data generation with various annotations for AI model training.

3. Why is multi-robot simulation important?
   - A: It reduces individual robot capabilities
   - B: It enables testing of coordination and communication systems
   - C: It makes robots move slower
   - D: It eliminates the need for programming

   **Answer: B** - Multi-robot simulation enables testing of coordination, communication, and fleet management systems.

4. What is the benefit of synthetic data generation in Isaac Sim?
   - A: It increases hardware costs
   - B: It provides labeled training data without real-world collection
   - C: It reduces computing power
   - D: It eliminates the need for sensors

   **Answer: B** - Synthetic data generation provides labeled training data without the time and cost of real-world data collection.

5. What does hardware-in-the-loop testing involve?
   - A: Testing without any hardware
   - B: Connecting real hardware to simulated environments
   - C: Testing only in simulation
   - D: Hardware that operates independently

   **Answer: B** - Hardware-in-the-loop testing connects real hardware components to simulated environments for validation.

## Learning Outcomes

After completing this chapter, students will be able to:
- Implement GPU-accelerated robotics systems
- Integrate AI perception and navigation capabilities
- Develop simulation-to-reality pipelines
- Optimize robot performance using NVIDIA platforms

## Prerequisites

- Basic understanding of Python programming
- Fundamentals of linear algebra and calculus
- Basic knowledge of robotics concepts
- Introduction to machine learning concepts
- Completion of Module 0 (Introduction and Foundations)
- Completion of Chapter 01 (Physical AI Basics)
- Completion of Chapter 03 (ROS2 Nodes, Topics & Services)
- Completion of Chapter 11 (Introduction to NVIDIA Isaac)
- Completion of Chapter 12 (Isaac SDK & APIs)

## Estimated Duration

6 hours