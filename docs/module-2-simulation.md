# Module 2: Simulation Environments - Gazebo & Unity 🎮

## Overview

Simulation is crucial for developing and testing humanoid robots before deploying to real hardware. This module covers two primary simulation environments: Gazebo (physics-focused) and Unity (visual/VR-focused).

## Gazebo Simulation

### Architecture

Gazebo is built on the **Open Dynamics Engine (ODE)** and provides realistic physics simulation:

```xml
<!-- Example robot model in URDF -->
<?xml version="1.0"?>
<robot name="simple_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### Key Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Physics Engine** | Realistic collision detection and response | Manipulation tasks |
| **Sensors Simulation** | Cameras, LIDAR, IMU, force/torque | Perception systems |
| **ROS Integration** | Direct communication with ROS nodes | Control system testing |
| **Plugins System** | Extensible functionality | Custom behaviors |

### Gazebo Workflows

```bash
# Launch Gazebo with a world file
gazebo worlds/willow.world

# Launch with ROS integration
roslaunch gazebo_ros empty_world.launch

# Launch with robot model
roslaunch gazebo_ros spawn_model.launch \
  -param robot_description \
  -urdf -model my_robot
```

## Unity Simulation

### Architecture

Unity provides a more visually appealing simulation environment with:
- High-fidelity graphics rendering
- VR/AR support
- Advanced lighting and materials
- Cross-platform deployment

### Integration with Robotics

Unity Robotics provides packages for:
- **ROS-TCP-Connector**: Communication with ROS/ROS2
- **ML-Agents**: Reinforcement learning training
- **Simulation Framework**: Physics and environment tools

### Unity Robotics Setup

```csharp
// Example Unity ROS connector script
using Unity.Robotics.ROSTCPConnector;
using UnityEngine;

public class ROSConnector : MonoBehaviour
{
    ROSConnection ros;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<Unity.Robotics.ROSTCPConnector.MessageTypes.std_msgs.String>("/chatter");
    }

    void SendROSMessage()
    {
        var message = new Unity.Robotics.ROSTCPConnector.MessageTypes.std_msgs.String();
        message.data = "Hello from Unity!";
        ros.Publish("/chatter", message);
    }
}
```

## Comparison Table

| Aspect | Gazebo | Unity |
|--------|--------|-------|
| **Physics Accuracy** | ⭐⭐⭐⭐⭐ High fidelity | ⭐⭐⭐⭐ Good |
| **Visual Quality** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐⭐ Excellent |
| **ROS Integration** | ⭐⭐⭐⭐⭐ Native | ⭐⭐⭐ Plugin-based |
| **Learning Curve** | ⭐⭐⭐ Steep | ⭐⭐⭐⭐ Moderate |
| **VR Support** | ⭐⭐ Limited | ⭐⭐⭐⭐⭐ Excellent |
| **Cost** | ⭐⭐⭐⭐⭐ Free | ⭐⭐ Free tier |

## Best Practices

### For Gazebo:
1. **Model Simplification**: Use simplified collision models for better performance
2. **Sensor Configuration**: Calibrate simulated sensors to match real hardware
3. **World Design**: Create realistic environments that match deployment scenarios
4. **Plugin Development**: Write custom plugins for specific robot behaviors

### For Unity:
1. **Performance Optimization**: Use LOD (Level of Detail) for complex scenes
2. **Physics Tuning**: Adjust physics parameters to match real-world behavior
3. **Asset Management**: Organize 3D models and materials efficiently
4. **Testing Scenarios**: Create diverse simulation environments

## Transitioning to Real Hardware

Both simulation environments should include:
- **Hardware-in-the-loop (HIL)** testing capabilities
- **Reality gap analysis** to understand simulation limitations
- **Transfer learning** techniques to bridge simulation-to-reality
- **Validation protocols** to ensure safe real-world deployment

## Advanced Topics

### Domain Randomization
- Randomizing simulation parameters to improve real-world transfer
- Varying lighting, textures, and physics parameters
- Training robust perception systems

### Multi-Robot Simulation
- Coordinating multiple robots in shared environments
- Communication and collaboration protocols
- Swarm robotics scenarios