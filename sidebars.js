/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      items: [
        {
          type: 'category',
          label: 'Module 1: Introduction to Physical AI',
          collapsible: true,
          collapsed: false,
          items: [
            '01-Chapter-1-Introduction',
            '02-Chapter-2-Physical-AI-Basics',
            '03-Chapter-3-Sensing-And-Embodied-Intelligence',
            '04-Chapter-4-ROS2-Nodes-Topics-Services'
          ],
        },
        {
          type: 'category',
          label: 'Module 2: ROS2 Communication & Simulation',
          collapsible: true,
          collapsed: false,
          items: [
            '05-Chapter-1-ROS2-Communication-Patterns',
            '06-Chapter-2-Practical-ROS2-Examples',
            '07-Chapter-3-Introduction-To-Digital-Twins',
            '08-Chapter-4-Gazebo-Simulation-Basics'
          ],
        },
        {
          type: 'category',
          label: 'Module 3: Digital Twins & Unity Integration',
          collapsible: true,
          collapsed: false,
          items: [
            '09-Chapter-1-Integrating-Unity-For-Visualization',
            '10-Chapter-2-Digital-Twin-Robotics-Examples',
            '11-Chapter-3-Best-Practices-Optimization'
          ],
        },
        {
          type: 'category',
          label: 'Module 4: Isaac SDK & ROS2 Integration',
          collapsible: true,
          collapsed: false,
          items: [
            '13-Chapter-1-Isaac-SDK-APIs',
            '14-Chapter-2-Isaac-Robot-Simulation-Examples',
            '15-Chapter-3-Integration-With-ROS2',
            '16-Chapter-4-Isaac-Best-Practices-Optimization'
          ],
        },
        {
          type: 'category',
          label: 'Module 5: Vision, Language & Action Concepts',
          collapsible: true,
          collapsed: false,
          items: [
            '17-Chapter-1-Vision-Language-Action-Concepts',
            '18-Chapter-2-Humanoid-Locomotion-Control',
            '19-Chapter-3-Vision-Based-Navigation-Examples',
            '20-Chapter-4-Language-Action-Integration'
          ],
        },
        {
          type: 'category',
          label: 'Module 6: Capstone Project',
          collapsible: true,
          collapsed: false,
          items: [
            '21-humanoid-project-examples-capstone'
          ],
        },
      ],
    },
  ],
};

export default sidebars;