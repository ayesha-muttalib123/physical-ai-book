# Module 1: ROS 2 - The Robot Operating System 🤖

## Overview

ROS 2 (Robot Operating System 2) is the foundation for modern robotics development. Unlike its predecessor, ROS 2 is built on DDS (Data Distribution Service) for robust, real-time communication between robot components.

## Core Concepts

| Component | Purpose | Example Usage |
|-----------|---------|---------------|
| **Nodes** | Individual processes performing computation | Perception node, Control node |
| **Topics** | Communication channels for data streams | Sensor data, motor commands |
| **Services** | Request/response communication | Calibration, configuration |
| **Actions** | Goal-oriented communication | Navigation, manipulation |
| **Parameters** | Configuration values | Motor limits, sensor settings |

## Architecture

ROS 2 uses a **distributed architecture** where nodes can run on different machines:

```bash
# Example ROS 2 workspace setup
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Key Features

### 1. Real-time Performance
- DDS-based communication ensures deterministic timing
- Support for real-time operating systems
- Priority-based message handling

### 2. Security
- Built-in authentication and encryption
- Secure communication protocols
- Access control mechanisms

### 3. Multi-platform Support
- Linux, Windows, macOS compatibility
- Cross-platform message definitions
- Consistent APIs across platforms

## Practical Implementation

### Creating a Simple Publisher

```cpp
// publisher_member_function.cpp
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

using namespace std::chrono_literals;

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, ROS 2 World! " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};
```

## ROS 2 Distributions

| Distribution | Code Name | Support Status |
|--------------|-----------|----------------|
| **Humble Hawksbill** | LTS (Long Term Support) | Active until 2027 |
| **Iron Irwini** | Standard | Active until 2025 |
| **Jazzy Jalisco** | Latest | Active until 2026 |

## Best Practices

1. **Package Organization**: Group related functionality into packages
2. **Launch Files**: Use launch files for complex system startup
3. **Parameter Management**: Externalize configuration parameters
4. **Testing**: Implement unit and integration tests
5. **Documentation**: Maintain clear API documentation

## Integration with Physical AI

ROS 2 serves as the communication backbone for Physical AI systems, enabling:
- Sensor data fusion from multiple modalities
- Coordinated control of complex robotic systems
- Real-time AI inference integration
- Distributed computing across robot components