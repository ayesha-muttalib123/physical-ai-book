import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/01-Chapter-1-Introduction">
            Start Reading - 5 min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="Complete guide to Physical AI & Humanoid Robotics">
      <HomepageHeader />
      <main>
        <section className={styles.modulesSection}>
          <div className="container padding-horiz--md">
            <h2 className={styles.modulesTitle}>Book Modules</h2>
            <p className={styles.modulesSubtitle}>Explore the comprehensive curriculum organized into structured modules</p>

            <div className="row">
              <div className="col col--4 margin-vert--md">
                <div className="card">
                  <div className="card__header">
                    <h3>Module 1: Introduction to Physical AI</h3>
                  </div>
                  <div className="card__body">
                    <p>
                      Foundations of Physical AI, basics of embodied intelligence, sensing technologies, and ROS2 fundamentals.
                    </p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/01-Chapter-1-Introduction">
                      Start Module
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--4 margin-vert--md">
                <div className="card">
                  <div className="card__header">
                    <h3>Module 2: ROS2 Communication & Simulation</h3>
                  </div>
                  <div className="card__body">
                    <p>
                      Deep dive into ROS2 communication patterns, practical examples, and introduction to digital twins.
                    </p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/05-Chapter-1-ROS2-Communication-Patterns">
                      Start Module
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--4 margin-vert--md">
                <div className="card">
                  <div className="card__header">
                    <h3>Module 3: Digital Twins & Unity Integration</h3>
                  </div>
                  <div className="card__body">
                    <p>
                      Explore digital twin concepts, Gazebo simulation, Unity visualization, and best practices.
                    </p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/09-Chapter-1-Integrating-Unity-For-Visualization">
                      Start Module
                    </Link>
                  </div>
                </div>
              </div>
            </div>

            <div className="row">
              <div className="col col--4 margin-vert--md">
                <div className="card">
                  <div className="card__header">
                    <h3>Module 4: Isaac SDK & ROS2 Integration</h3>
                  </div>
                  <div className="card__body">
                    <p>
                      NVIDIA Isaac SDK APIs, robot simulation examples, and integration with ROS2 systems.
                    </p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/13-Chapter-1-Isaac-SDK-APIs">
                      Start Module
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--4 margin-vert--md">
                <div className="card">
                  <div className="card__header">
                    <h3>Module 5: Vision, Language & Action Concepts</h3>
                  </div>
                  <div className="card__body">
                    <p>
                      Vision-language-action integration, humanoid locomotion control, and navigation systems.
                    </p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/17-Chapter-1-Vision-Language-Action-Concepts">
                      Start Module
                    </Link>
                  </div>
                </div>
              </div>

              <div className="col col--4 margin-vert--md">
                <div className="card">
                  <div className="card__header">
                    <h3>Module 6: Capstone Project</h3>
                  </div>
                  <div className="card__body">
                    <p>
                      Comprehensive project examples bringing together all concepts learned throughout the book.
                    </p>
                  </div>
                  <div className="card__footer">
                    <Link
                      className="button button--primary button--block"
                      to="/docs/21-humanoid-project-examples-capstone">
                      Start Module
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </Layout>
  );
}