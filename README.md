# Physical World Intelligence System

Multi-modal AI system for understanding 3D space, materials, and physics from camera input.

## 🎯 Goal
Help people make decisions about physical spaces:
- Will furniture fit through doorways?
- Is this wall load-bearing?
- Can I safely mount objects here?

## 🛠️ Tech Stack
- Python 3.14
- OpenCV (Computer Vision)
- PyTorch (Deep Learning)
- MiDaS (Depth Estimation)
- NumPy, Matplotlib, Pillow

## 📅 Development Timeline
- **Week 1**: Depth estimation from images ✅ (Day 1 complete!)
- **Week 2-4**: 3D reconstruction & object detection
- **Month 2**: Material understanding + audio analysis
- **Month 3**: Physics simulation + safety checking
- **Month 4**: Full deployment with Docker

## 🚀 Current Progress

### Week 1 - Day 1 ✅
- [x] Set up development environment
- [x] Configured Git & GitHub
- [x] Installed computer vision libraries (OpenCV, PyTorch)
- [x] Implemented camera capture system
- [x] **Built depth estimation AI using MiDaS model**
- [x] Generated first depth map from camera input

**First Result:**
![Depth Estimation Result](my_first_depth_map.png)

The depth map shows spatial understanding - warm colors (red/yellow) represent close objects, cool colors (blue/purple) represent distant objects.

## 🧠 What I'm Learning

**Day 1 Learnings:**
- How to set up professional Python development environments
- Git workflow and version control best practices
- Computer vision fundamentals (images as arrays)
- Running pre-trained neural networks with PyTorch
- Depth estimation from monocular images
- Debugging dependency errors in AI projects

## 🔧 Setup Instructions
```bash
# Clone repository
git clone https://github.com/SurajMadhavarapu/physical-world-ai.git
cd physical-world-ai

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Run the Project

**Test camera:**
```bash
python test_camera.py
```

**Generate depth map:**
```bash
python depth_estimation_simple.py
# Press SPACE to capture, ESC to cancel
```

## 📁 Project Structure
```
physical-world-ai/
├── test_camera.py              # Camera capture test
├── depth_estimation_simple.py  # Depth estimation from camera
├── requirements.txt            # Python dependencies
├── my_first_depth_map.png     # Example output
└── README.md                   # This file
```

## 🎯 Next Steps
- Process video streams for continuous depth estimation
- Build 3D point cloud from depth maps
- Implement object detection (furniture, walls, doors)
- Develop distance measurement system

## 📝 Development Log

**2025-02-14:** Project initialization. Successfully implemented depth estimation AI. First milestone achieved - can now estimate relative depth from 2D camera images using MiDaS neural network.

---

**Status:** 🟢 Active Development  
**Day:** 1 of ~120  
**Progress:** Foundation complete, moving to spatial understanding

Built with 🧠 as a learning journey from zero to production-ready AI system.