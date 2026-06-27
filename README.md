Intelligent-Free-Left-Turn-Management-System
An AI-powered traffic management system designed to optimize free-left turn efficiency, reduce congestion, and enhance road safety using real-time vehicle detection, dataset analysis, and adaptive signal control.

📌 Overview

Urban intersections often suffer from inefficiencies due to poorly managed free-left turns, leading to congestion and safety risks.

This project solves that by combining:

Computer Vision (YOLOv8) for real-time vehicle detection
Data Analysis from traffic datasets (PDF reports)
Intelligent Signal Control based on real-time + historical insights
Adaptive Decision Engine for dynamic traffic signal management
Features

Real-Time Monitoring
Detects vehicles using YOLOv8 Supports:
Live camera feed
Uploaded video files
Identifies:
Vehicles in free-left lane
Blocking vehicles (duration-based)
Dataset Analysis

Upload traffic PDFs
Extract:
Violations
Traffic volume
Pedestrian data
Generate:
Risk assessment
Peak hours
Smart recommendations
Intelligent Signal Control

Dynamic switching between: 🟡 Free Left 🟢 Protected Left 🔴 All Red
Decision based on:
Number of violations
Blocking duration
Peak hours
Historical dataset insights
Analytics Dashboard

Signal phase transitions
Violation trends by hour
Event logs
Risk visualization
Report Generation

Download PDF reports with:
Analysis summary
Risk levels
Recommendations
Project Structure ├── app.py # Main Streamlit application ├── core/ │ ├── detector.py # YOLO-based vehicle detection │ ├── signal_controller.py # Decision engine for signals │ ├── data_analyzer.py # Dataset analysis & insights │ ├── pdf_parser.py # PDF data extraction │ ├── assets/ # (optional) sample videos/images ├── requirements.txt └── README.md

⚙️ Tech Stack

Frontend/UI: Streamlit
Computer Vision: OpenCV, YOLOv8 (Ultralytics)
Data Processing: Pandas
Visualization: Plotly
PDF Parsing: pdfplumber
Backend Logic: Python
Installation 1️⃣ Clone the Repository git clone https://github.com/spacedragon51/Free-Left-Turn-Management-System.git cd free-left-turn-ai 2️⃣ Create Virtual Environment python -m venv venv source venv/bin/activate # Windows: venv\Scripts\activate 3️⃣ Install Dependencies pip install -r requirements.txt ▶️ Usage Run the Application : streamlit run app.py

How It Works

Real-Time Mode
Select:
Camera OR upload video
Define free-left lane region
Start detection
System:
Detects vehicles
Tracks blocking duration
Adjusts signal dynamically
Dataset Mode

Upload traffic PDF
Click Analyze
System: - Extracts data - Computes risk score - Suggests improvements
Core Logic

Vehicle Detection
Uses YOLOv8 to detect:
Cars
Bikes
Buses
Trucks
Tracks vehicles across frames
Identifies lane violations
Pedestrian crossing detection
Automated zebra crossing detection
Pedestrian waiting area analysis
Signal Decision Engine

Switches to Protected Left when:
Violations exceed threshold
Vehicles block lane for long duration
Peak hour sensitivity triggers
Pedestrian crossing the road
Risk Assessment

Based on:
Total violations
Peak congestion hours
Outputs:
LOW / MEDIUM / HIGH risk
📌 Use Cases

Smart city traffic systems
Urban intersection optimization
Traffic police decision support
Government traffic analytics
Hackathon / research projects
Future Improvements

Multi-camera intersection monitoring
Edge deployment (Jetson Nano / Raspberry Pi)
Integration with real traffic signals
AI-based predictive traffic flow
Mobile app for traffic authorities
