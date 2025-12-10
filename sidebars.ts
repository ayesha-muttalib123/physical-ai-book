import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar configuration for the Physical AI & Humanoid Robotics book
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      items: [
        'introduction',
        {
          type: 'category',
          label: 'Core Modules',
          items: [
            'module-1-ros2',
            'module-2-simulation',
            'module-3-nvidia-isaac',
            
          ]
        },
        {
          type: 'category',
          label: 'Course Structure',
          items: [
            'learning-outcomes',
            'weekly-breakdown',
            'assessments'
          ]
        },
        {
          type: 'category',
          label: 'Hardware & Implementation',
          items: [
            'hardware-workstation',
            'hardware-edge-kit',
            'hardware-robot-options',
            'economy-jetson-kit',
            'cloud-alternative'
          ]
        },
        'future-of-humanoids'
      ]
    }
  ]
};

export default sidebars;
