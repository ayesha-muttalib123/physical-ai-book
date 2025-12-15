#!/usr/bin/env python3
"""
Auto-append PHR (Project Health Records) script
Automatically updates PHR files when changes are detected in the system
"""

import os
import yaml
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any


class PHRAutoAppender:
    def __init__(self, config_path: str = ".specify/config.yml"):
        """Initialize the PHR auto-appender with configuration"""
        self.config = self.load_config(config_path)
        self.phr_dir = Path(self.config['directories']['phr_folder'])
        self.tasks_dir = Path(self.config['directories']['tasks_folder'])
        self.plans_dir = Path(self.config['directories']['plans_folder'])

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_latest_task_status(self) -> Dict[str, Any]:
        """Get the latest status from task files"""
        task_status = {}

        for task_file in self.tasks_dir.glob("*.yml"):
            with open(task_file, 'r') as f:
                task_data = yaml.safe_load(f)

            if 'tasks' in task_data:
                completed = sum(1 for task in task_data['tasks'] if task.get('status') == 'completed')
                total = len(task_data['tasks'])
                task_status[task_data.get('task_id', task_file.stem)] = {
                    'completed': completed,
                    'total': total,
                    'progress': completed / total if total > 0 else 0
                }

        return task_status

    def get_latest_plan_status(self) -> Dict[str, Any]:
        """Get the latest status from plan files"""
        plan_status = {}

        for plan_file in self.plans_dir.glob("*.yml"):
            with open(plan_file, 'r') as f:
                plan_data = yaml.safe_load(f)

            plan_id = plan_data.get('plan_id', plan_file.stem)
            plan_status[plan_id] = {
                'status': plan_data.get('status', 'unknown'),
                'progress': plan_data.get('progress_percentage', 0) if 'progress_percentage' in plan_data else 0,
                'last_updated': plan_data.get('last_updated', str(datetime.datetime.now()))
            }

        return plan_status

    def update_project_health_overview(self):
        """Update the main project health overview file"""
        phr_file = self.phr_dir / "project-health-overview.yml"

        if phr_file.exists():
            with open(phr_file, 'r') as f:
                phr_data = yaml.safe_load(f)
        else:
            phr_data = {
                'phr_id': 'project-health-overview',
                'title': 'Physical AI & Humanoid Robotics - Project Health Overview',
                'description': 'Comprehensive health assessment of the Physical AI & Humanoid Robotics project',
                'version': '1.0',
                'created_at': str(datetime.datetime.now()),
                'last_updated': str(datetime.datetime.now()),
                'status': 'active',
                'health_metrics': {},
                'module_health': [],
                'component_health': [],
                'risk_assessment': {},
                'performance_metrics': {},
                'recommendations': [],
                'compliance_status': {},
                'next_milestones': []
            }

        # Update task-based progress
        task_status = self.get_latest_task_status()
        total_completed = sum(status['completed'] for status in task_status.values())
        total_tasks = sum(status['total'] for status in task_status.values())

        if total_tasks > 0:
            phr_data['health_metrics']['progress_percentage'] = round((total_completed / total_tasks) * 100, 2)

        # Update timestamp
        phr_data['last_updated'] = str(datetime.datetime.now())

        # Update module health based on task progress
        for module in phr_data.get('module_health', []):
            module_id = module['module_id']
            # Find corresponding task status
            for task_id, status in task_status.items():
                if module_id.replace('-', '_') in task_id or task_id.startswith(module_id.split('-')[0]):
                    module['progress'] = status['progress'] * 100
                    module['health_status'] = 'green' if status['progress'] >= 0.9 else 'yellow' if status['progress'] >= 0.7 else 'red'
                    module['last_activity'] = str(datetime.datetime.now())
                    break

        # Write updated PHR data
        with open(phr_file, 'w') as f:
            yaml.dump(phr_data, f, default_flow_style=False, allow_unicode=True)

        print(f"Updated project health overview at {phr_file}")

    def update_specific_health_files(self):
        """Update specific health files based on related task and plan files"""
        # Update ROS2 architecture health
        self.update_ros2_health()

        # Update Isaac platform health
        self.update_isaac_health()

    def update_ros2_health(self):
        """Update ROS2 architecture health based on related tasks"""
        phr_file = self.phr_dir / "ros2-architecture-health.yml"

        if phr_file.exists():
            with open(phr_file, 'r') as f:
                phr_data = yaml.safe_load(f)

            # Update based on related task files
            task_status = self.get_latest_task_status()

            # Find ROS2-related tasks and update component health
            for component in phr_data.get('component_health', []):
                comp_id = component['component_id']

                # Update based on corresponding task progress
                for task_id, status in task_status.items():
                    if comp_id in task_id:
                        component['implementation_status'] = 'complete' if status['progress'] == 1.0 else 'in_progress'
                        component['last_updated'] = str(datetime.datetime.now())
                        break

            # Update timestamp
            phr_data['last_updated'] = str(datetime.datetime.now())

            # Write updated data
            with open(phr_file, 'w') as f:
                yaml.dump(phr_data, f, default_flow_style=False, allow_unicode=True)

            print(f"Updated ROS2 architecture health at {phr_file}")

    def update_isaac_health(self):
        """Update Isaac platform health based on related tasks"""
        phr_file = self.phr_dir / "isaac-platform-health.yml"

        if phr_file.exists():
            with open(phr_file, 'r') as f:
                phr_data = yaml.safe_load(f)

            # Update based on related task files
            task_status = self.get_latest_task_status()

            # Find Isaac-related tasks and update component health
            for component in phr_data.get('component_health', []):
                comp_id = component['component_id']

                # Update based on corresponding task progress
                for task_id, status in task_status.items():
                    if comp_id in task_id:
                        component['implementation_status'] = 'complete' if status['progress'] == 1.0 else 'in_progress'
                        component['last_updated'] = str(datetime.datetime.now())
                        break

            # Update timestamp
            phr_data['last_updated'] = str(datetime.datetime.now())

            # Write updated data
            with open(phr_file, 'w') as f:
                yaml.dump(phr_data, f, default_flow_style=False, allow_unicode=True)

            print(f"Updated Isaac platform health at {phr_file}")

    def run_auto_append(self):
        """Run the auto-append process"""
        print("Running PHR auto-append process...")

        # Update the main project health overview
        self.update_project_health_overview()

        # Update specific health files
        self.update_specific_health_files()

        print("PHR auto-append process completed successfully.")


def main():
    """Main function to run the auto-append process"""
    appender = PHRAutoAppender()
    appender.run_auto_append()


if __name__ == "__main__":
    main()