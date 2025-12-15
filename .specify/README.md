# Specify Memory Structure

This directory contains the organized structure for the Physical AI & Humanoid Robotics project, organized according to the specification.

## Directory Structure

```
.specify/
├── memory/
│   ├── tasks/           # Task files derived from specifications
│   ├── plans/           # Planning files for project implementation
│   └── phr/             # Project Health Records (automatically updated)
├── config.yml          # Configuration for the organization system
├── scripts/
│   └── auto-append-phr.py  # Script for auto-updating PHR files
└── README.md           # This file
```

## Auto-Append Functionality

The system includes auto-append functionality that automatically updates Project Health Records (PHR) files when changes occur in tasks or plans. This is implemented through:

1. **Configuration**: The `.specify/config.yml` file defines the auto-append behavior
2. **Script**: The `auto-append-phr.py` script processes updates to PHR files
3. **Hooks**: The system can be configured to run updates automatically

### Running the Auto-Append Process

To manually run the auto-append process:

```bash
python .specify/scripts/auto-append-phr.py
```

This script will:
- Read the latest status from all task files
- Update the corresponding PHR files with current progress
- Update health metrics based on implementation status
- Maintain timestamps and other metadata

## Files Organization

### Tasks
- Located in `.specify/memory/tasks/`
- Each task file corresponds to a specification chapter
- Contains actionable items derived from the specifications

### Plans
- Located in `.specify/memory/plans/`
- Each plan file contains project planning information
- Includes timelines, resources, and implementation strategies

### PHR (Project Health Records)
- Located in `.specify/memory/phr/`
- Automatically updated with implementation progress
- Track health status of different components
- Include performance metrics and risk assessments

## Maintaining Structure

The system maintains structure through:
- Consistent naming conventions
- Hierarchical organization
- Automated updates to health records
- Configuration-based organization rules