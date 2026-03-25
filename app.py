#!/usr/bin/env python3
"""
Intelligent Free Left Turn Management System
Complete Scratch-Built Application with Pedestrian Safety
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import tempfile
import sys
from typing import Optional

# Page config
st.set_page_config(
    page_title="Intelligent Free Left Turn Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-free {
        background-color: #ffa500;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .signal-protected {
        background-color: #00cc66;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .signal-pedestrian {
        background-color: #ff4b4b;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-low {
        background-color: #00cc66;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
from core.detector import UnifiedDetector, DetectionType
from core.signal_controller import EnhancedSignalController, SignalPhase
from core.pdf_parser import PDFParser
from core.data_analyzer import DataAnalyzer


class AppState:
    """Application state management"""
    def __init__(self):
        self.detector = None
        self.controller = EnhancedSignalController()
        self.video_source = None
        self.is_running = False
        self.analysis_results = None
        self.intersection_name = "Banashankari Junction"
        self.lane_region = None
        self.crossing_region = None


def init_session_state():
    """Initialize session state"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False


def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚦 Intelligent Free Left Turn Management System</h1>
        <p>AI-powered traffic management | Vehicle Detection | Pedestrian Safety | Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)


def render_region_config():
    """Render lane and crossing region configuration"""
    st.subheader("🎯 Region Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Free-Left Lane Region**")
        st.info("Define the area where turning vehicles wait")
        
        lane_x1 = st.number_input("Lane Top-Left X", 0, 1280, 100, key="lane_x1")
        lane_y1 = st.number_input("Lane Top-Left Y", 0, 720, 400, key="lane_y1")
        lane_x2 = st.number_input("Lane Bottom-Right X", 0, 1280, 550, key="lane_x2")
        lane_y2 = st.number_input("Lane Bottom-Right Y", 0, 720, 450, key="lane_y2")
        
        lane_region = [(lane_x1, lane_y1), (lane_x2, lane_y1), (lane_x2, lane_y2), (lane_x1, lane_y2)]
    
    with col2:
        st.markdown("**Pedestrian Crossing Region**")
        st.info("Define the area where pedestrians cross")
        
        cross_x1 = st.number_input("Crossing Top-Left X", 0, 1280, 300, key="cross_x1")
        cross_y1 = st.number_input("Crossing Top-Left Y", 0, 720, 300, key="cross_y1")
        cross_x2 = st.number_input("Crossing Bottom-Right X", 0, 1280, 800, key="cross_x2")
        cross_y2 = st.number_input("Crossing Bottom-Right Y", 0, 720, 500, key="cross_y2")
        
        crossing_region = [(cross_x1, cross_y1), (cross_x2, cross_y1), (cross_x2, cross_y2), (cross_x1, cross_y2)]
    
    # Preview
    preview = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.polylines(preview, [np.array(lane_region, dtype=np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(preview, [np.array(crossing_region, dtype=np.int32)], True, (255, 255, 0), 2)
    cv2.putText(preview, "FREE LEFT LANE", (lane_x1, lane_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(preview, "PEDESTRIAN CROSSING", (cross_x1, cross_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    st.image(preview, caption="Region Preview", use_container_width=True)
    
    if st.button("Apply Regions", type="primary"):
        st.session_state.app_state.lane_region = lane_region
        st.session_state.app_state.crossing_region = crossing_region
        st.success("✅ Regions applied!")
    
    return lane_region, crossing_region


def render_video_source():
    """Render video source selection"""
    st.subheader("📹 Video Source")
    
    source_type = st.radio(
        "Source Type",
        ["📷 Camera", "🎬 Video File"],
        horizontal=True
    )
    
    if source_type == "📷 Camera":
        camera_id = st.selectbox("Camera", ["Webcam (0)", "External (1)"], index=0)
        source = int(camera_id.split("(")[1].split(")")[0])
        return {'type': 'camera', 'source': source}
    else:
        uploaded = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded:
            temp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            return {'type': 'video', 'source': temp_path, 'name': uploaded.name}
        return None


def render_signal_status():
    """Render current signal status"""
    app = st.session_state.app_state
    state = app.controller.get_state()
    
    st.subheader("🚥 Signal Status")
    
    phase = state['phase']
    
    if "FREE" in phase:
        st.markdown(f"""
        <div class="signal-free">
            <h2>{phase}</h2>
            <p>Yield to oncoming traffic</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PROTECTED" in phase:
        st.markdown(f"""
        <div class="signal-protected">
            <h2>{phase}</h2>
            <p>Green arrow - Safe to turn</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PEDESTRIAN" in phase:
        st.markdown(f"""
        <div class="signal-pedestrian">
            <h2>{phase}</h2>
            <p>Stop for pedestrians</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚗 Blocking Vehicles", state['blocking_vehicles'])
    with col2:
        st.metric("🚶 Pedestrians Waiting", state.get('pedestrians_waiting', 0))
    with col3:
        st.metric("⏱️ Phase Duration", f"{state['phase_duration']:.0f}s")
    with col4:
        risk = state['risk_level']
        st.metric("📊 Risk Level", risk)


def render_realtime_feed():
    """Render real-time detection feed"""
    app = st.session_state.app_state
    
    if not app.is_running or not app.video_source:
        st.info("No active source. Configure regions and click START.")
        return
    
    # Read frame
    ret, frame = app.video_source.read()
    if not ret:
        st.warning("End of video or no frame")
        return
    
    # Detect
    detections, annotated = app.detector.detect(frame)
    
    # Get lane vehicles and pedestrians
    lane_vehicles = [d for d in detections if d.type == DetectionType.VEHICLE and d.in_lane]
    pedestrians = [d for d in detections if d.type == DetectionType.PEDESTRIAN and d.at_crossing]
    
    # Update controller
    phase = app.controller.update_detections(lane_vehicles, pedestrians)
    
    # Display
    st.image(annotated, channels="BGR", use_container_width=True)
    
    # Detection info
    col1, col2 = st.columns(2)
    with col1:
        if lane_vehicles:
            st.warning(f"⚠️ {len(lane_vehicles)} vehicle(s) in free-left lane")
            for v in lane_vehicles[:3]:
                st.write(f"• {v.class_name} - {v.stationary_time:.1f}s")
        else:
            st.success("✅ Free-left lane clear")
    
    with col2:
        if pedestrians:
            st.info(f"🚶 {len(pedestrians)} pedestrian(s) at crossing")
            for p in pedestrians[:3]:
                st.write(f"• Waiting to cross")
        else:
            st.success("✅ No pedestrians")
    
    # Auto-refresh
    time.sleep(0.03)
    st.rerun()


def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.header("🎮 Controls")
        
        mode = st.radio(
            "Mode",
            ["📊 Dataset Analysis", "📹 Real-time Monitoring"]
        )
        
        st.divider()
        
        if mode == "📹 Real-time Monitoring":
            # Region config
            lane_region, crossing_region = render_region_config()
            
            st.divider()
            
            # Source selection
            source = render_video_source()
            
            # Start/Stop
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ START", type="primary", use_container_width=True):
                    if source:
                        app = st.session_state.app_state
                        app.detector = UnifiedDetector(
                            lane_region=lane_region,
                            crossing_region=crossing_region,
                            conf_threshold=0.4
                        )
                        
                        if source['type'] == 'camera':
                            app.video_source = cv2.VideoCapture(source['source'])
                        else:
                            app.video_source = cv2.VideoCapture(source['source'])
                        
                        if app.video_source.isOpened():
                            app.is_running = True
                            st.success("Started!")
                            st.rerun()
                        else:
                            st.error("Cannot open source")
            
            with col2:
                if st.button("⏹️ STOP", use_container_width=True):
                    app = st.session_state.app_state
                    app.is_running = False
                    if app.video_source:
                        app.video_source.release()
                    st.info("Stopped")
            
            st.divider()
            
            # Manual override
            st.subheader("🔧 Manual Override")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔒 Protected Left"):
                    app = st.session_state.app_state
                    app.controller.manual_protect()
                    st.success("Protected left activated")
            with col2:
                if st.button("🚶 Pedestrian Mode"):
                    app = st.session_state.app_state
                    app.controller.manual_pedestrian()
                    st.success("Pedestrian mode activated")
            
            if st.button("🔄 Reset to Auto"):
                app = st.session_state.app_state
                app.controller.manual_reset()
                st.info("Auto mode restored")
        
        else:
            # Dataset analysis mode
            st.subheader("📄 Upload Dataset")
            
            uploaded = st.file_uploader("Upload PDF", type=['pdf'])
            
            if uploaded:
                if st.button("Analyze", type="primary"):
                    with st.spinner("Analyzing..."):
                        temp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
                        with open(temp_path, 'wb') as f:
                            f.write(uploaded.getbuffer())
                        
                        parser = PDFParser()
                        data = parser.parse(temp_path)
                        analyzer = DataAnalyzer(data)
                        analysis = analyzer.analyze()
                        
                        st.session_state.app_state.analysis_results = analysis
                        st.session_state.app_state.controller.integrate_dataset(analysis)
                        st.session_state.analysis_complete = True
                        st.success("Analysis complete!")
        
        return mode


def render_dataset_results():
    """Render dataset analysis results"""
    results = st.session_state.app_state.analysis_results
    if not results:
        return
    
    st.subheader("📊 Analysis Results")
    
    risk = results.get('risk_assessment', {})
    risk_level = risk.get('level', 'UNKNOWN')
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h3>Risk Assessment: {risk_level} (Score: {risk.get('score', 0)}/100)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Violations", results.get('violation_analysis', {}).get('total_violations', 0))
    with col2:
        st.metric("Risk Score", f"{risk.get('score', 0)}/100")
    with col3:
        peak = results.get('violation_analysis', {}).get('peak_hours', {})
        if peak:
            st.metric("Peak Hours", ', '.join([f"{h}:00" for h in peak.keys()]))
    
    recs = results.get('recommendations', [])
    if recs:
        st.markdown("### 💡 Recommendations")
        for rec in recs[:5]:
            st.info(rec)


def render_analytics():
    """Render analytics dashboard"""
    app = st.session_state.app_state
    events = app.controller.get_events(50)
    
    if not events:
        return
    
    st.subheader("📈 Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Event distribution
        event_counts = {}
        for e in events:
            event_counts[e['event']] = event_counts.get(e['event'], 0) + 1
        
        if event_counts:
            df = pd.DataFrame(list(event_counts.items()), columns=['Event', 'Count'])
            fig = px.pie(df, values='Count', names='Event', title="Event Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Timeline
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        hourly = df['hour'].value_counts().sort_index()
        
        if not hourly.empty:
            fig = go.Figure(data=[go.Bar(x=hourly.index.astype(str), y=hourly.values)])
            fig.update_layout(title="Events by Hour")
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 Event Log"):
        for event in reversed(events[-20:]):
            st.write(f"**{event['timestamp'][:19]}** - {event['event']}")


def main():
    """Main application entry point"""
    init_session_state()
    render_header()
    
    mode = render_sidebar()
    
    if mode == "📹 Real-time Monitoring":
        col1, col2 = st.columns([3, 2])
        
        with col1:
            render_realtime_feed()
        
        with col2:
            render_signal_status()
            render_analytics()
    
    else:
        render_dataset_results()
        render_analytics()
    
    st.divider()
    st.caption(f"🕒 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System: Active")


if __name__ == "__main__":
    main()