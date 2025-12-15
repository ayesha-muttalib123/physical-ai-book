import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import Heading from '@theme/Heading';
import styles from './styles.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/01-Chapter-1-Introduction">
            Read the Book - 10 min ⏱️
          </Link>
        </div>
        <div className={clsx('margin-top--lg', styles.chapterLinks)}>
          <h3>Chapters:</h3>
          <div className="container">
            <div className="row">
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/01-Chapter-1-Introduction">Chapter 1: Introduction</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/02-Chapter-2-Physical-AI-Basics">Chapter 2: Physical AI Basics</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/03-Chapter-3-Sensing-And-Embodied-Intelligence">Chapter 3: Sensing and Embodied Intelligence</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/04-Chapter-4-ROS2-Nodes-Topics-Services">Chapter 4: ROS2 Nodes, Topics, and Services</Link>
              </div>
            </div>
            <div className="row">
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/04-Chapter-1-ROS2-Communication-Patterns">Chapter 1: ROS2 Communication Patterns</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/05-Chapter-2-Practical-ROS2-Examples">Chapter 2: Practical ROS2 Examples</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/06-Chapter-3-Introduction-To-Digital-Twins">Chapter 3: Introduction To Digital Twins</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/07-Chapter-4-Gazebo-Simulation-Basics">Chapter 4: Gazebo Simulation Basics</Link>
              </div>
            </div>
            <div className="row">
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/08-Chapter-1-Integrating-Unity-For-Visualization">Chapter 1: Integrating Unity For Visualization</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/09-Chapter-2-Digital-Twin-Robotics-Examples">Chapter 2: Digital Twin Robotics Examples</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/10-Chapter-3-Best-Practices-Optimization">Chapter 3: Best Practices & Optimization</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/11-Chapter-4-Introduction-To-NVIDIA-Isaac">Chapter 4: Introduction To NVIDIA Isaac</Link>
              </div>
            </div>
            <div className="row">
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/12-Chapter-1-Isaac-SDK-APIs">Chapter 1: Isaac SDK & APIs</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/13-Chapter-2-Isaac-Robot-Simulation-Examples">Chapter 2: Isaac Robot Simulation Examples</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/14-Chapter-3-Integration-With-ROS2">Chapter 3: Integration With ROS2</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/15-Chapter-4-Isaac-Best-Practices-Optimization">Chapter 4: Isaac Best Practices & Optimization</Link>
              </div>
            </div>
            <div className="row">
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/16-Chapter-1-Vision-Language-Action-Concepts">Chapter 1: Vision-Language-Action Concepts</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/17-Chapter-2-Humanoid-Locomotion-Control">Chapter 2: Humanoid Locomotion & Control</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/18-Chapter-3-Vision-Based-Navigation-Examples">Chapter 3: Vision-Based Navigation Examples</Link>
              </div>
              <div className="col col--3 margin-bottom--md">
                <Link className="button button--primary" to="/docs/19-Chapter-4-Language-Action-Integration">Chapter 4: Language-Action Integration</Link>
              </div>
            </div>
            <div className="row">
              <div className="col col--4 margin-bottom--md">
                <Link className="button button--primary" to="/docs/20-Chapter-1-Humanoid-Project-Examples-Capstone">Chapter 1: Humanoid Project Examples & Capstone</Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}