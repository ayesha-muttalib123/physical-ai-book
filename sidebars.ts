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
  tutorialSidebar: [
    'introduction',
    'module-1-ros2',
    'module-2-simulation',
    'module-3-nvidia-isaac',
    'module-4-vla',
    'learning-outcomes',
    'weekly-breakdown',
    'assessments',
    'hardware-workstation',
    'hardware-edge-kit',
    'hardware-robot-options',
    'cloud-alternative',
    'economy-jetson-kit',
    'future-of-humanoids'
  ],
};

export default sidebars;