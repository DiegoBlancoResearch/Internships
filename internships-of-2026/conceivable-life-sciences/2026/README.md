# Robotics & Computer Vision Internship
**Conceivable Life Sciences — June 2026**

## Overview

During a my research internship at Conceivable Life Sciences, 
I designed and implemented an end-to-end pipeline integrating industrial 
robotics, real-time computer vision, and reinforcement learning. The goal 
was to build an autonomous system capable of perceiving its environment 
through a camera, making strategic decisions, and executing precise 
physical movements using a Meca500 industrial robot arm.

## What I Built

### 1. Robot Control System
Programmed a Meca500 industrial robot arm using Python, implementing 
precise 6-axis movement control (x, y, z, α, β, γ). Built a Plotter 
class handling pen-up/pen-down mechanics, safe height transitions, 
and arbitrary trajectory execution.

### 2. Real-Time Computer Vision Pipeline
Built a full computer vision pipeline processing live camera feed from 
an industrial Huatang IP camera:
- Grayscale conversion and adaptive thresholding
- Morphological operations (erosion, closing) for noise reduction
- Probabilistic Hough Line Transform for line detection
- Angle-based classification of detected line segments
- Perspective correction (bird's-eye view transform) for geometric accuracy
- Stable line position computation via frame-averaged mean endpoints

### 3. Q-Learning Agent
Implemented a Q-Learning reinforcement learning agent from scratch 
that learns to play tic-tac-toe through self-play, without being 
explicitly programmed with strategy. The agent represents board states 
as immutable tuples, maintains a Q-table mapping (state, action) pairs 
to learned values, and improves its policy through exploration and 
exploitation over thousands of training episodes.

### 4. 3D-Printed Board Integration
Collaborated with the engineering team on a 3D-printed tic-tac-toe 
board with standardized geometry, enabling consistent and reliable 
computer vision detection across sessions.

## Technical Stack

- **Language:** Python 3
- **Computer Vision:** OpenCV, NumPy
- **Machine Learning:** Q-Learning (custom implementation)
- **Hardware:** Meca500 industrial robot arm, Huatang industrial IP camera
- **Dependencies:** exploration_utils (proprietary Conceivable Life Sciences 
  library — not included)

## Key Engineering Decisions

- Used probabilistic Hough Line Transform over standard Hough Lines 
  for segment-level detection, enabling angle-based classification
- Implemented frame-averaged mean endpoints rather than single-frame 
  detection to achieve temporally stable line output
- Applied perspective transform calibrated once per session, 
  eliminating geometric distortion from fixed camera angle
- Chose Q-Learning over Minimax to demonstrate genuine machine 
  learning rather than deterministic game-tree search

## Context

Internship at Conceivable Life Sciences, Mexico City — June 2026  
Supervisor: Gerardo Mendizabal and Estefanía  
CMO: Dr. Alejandro Chávez-Badiola
