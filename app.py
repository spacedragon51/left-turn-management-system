#!/usr/bin/env python3
"""
Intelligent Free Left Turn Management System
Complete Application with Auto Zebra Crossing Detection
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
import json
import base64
import io
from typing import Optional, Dict, List, Tuple
import logging

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
        animation: pulse 2s infinite;
    }
    .signal-protected {
        background-color: #00cc66;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        animation: pulse 1s infinite;
    }
    .signal-pedestrian {
        background-color: #ff4b4b;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        animation: blink 1s infinite;
    }
    .signal-emergency {
        background-color: #8b0000;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        animation: blink 0.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
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
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .status-active {
        background-color: #00cc66;
        color: white;
    }
    .status-warning {
        background-color: #ffa500;
        color: white;
    }
    .status-danger {
        background-color: #ff4b4b;
        color: white;
    }
    .crossing-detected {
        border: 2px solid #00cc66;
        background-color: rgba(0, 204, 102, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
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
        self.auto_detect_crossing = True
        self.crossing_detected = False
        self.stats_history = []
        self.video_file_path = None
        self.uploaded_pdf_path = None


def init_session_state():
    """Initialize session state"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'events' not in st.session_state:
        st.session_state.events = []


def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚦 Intelligent Free Left Turn Management System</h1>
        <p>AI-powered traffic management | Auto Zebra Detection | Pedestrian Safety | Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)


def render_manual_region_config():
    """Render manual region configuration"""
    st.subheader("🎯 Manual Region Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Free-Left Lane Region**")
        lane_x1 = st.number_input("Lane Left X", 0, 1280, 100, key="lane_x1")
        lane_y1 = st.number_input("Lane Top Y", 0, 720, 400, key="lane_y1")
        lane_x2 = st.number_input("Lane Right X", 0, 1280, 550, key="lane_x2")
        lane_y2 = st.number_input("Lane Bottom Y", 0, 720, 450, key="lane_y2")
        
        lane_region = [(lane_x1, lane_y1), (lane_x2, lane_y1), (lane_x2, lane_y2), (lane_x1, lane_y2)]
    
    with col2:
        st.markdown("**Pedestrian Crossing Region (Manual)**")
        st.info("If auto-detection is off, define crossing area manually")
        cross_x1 = st.number_input("Crossing Left X", 0, 1280, 300, key="cross_x1")
        cross_y1 = st.number_input("Crossing Top Y", 0, 720, 300, key="cross_y1")
        cross_x2 = st.number_input("Crossing Right X", 0, 1280, 800, key="cross_x2")
        cross_y2 = st.number_input("Crossing Bottom Y", 0, 720, 500, key="cross_y2")
        
        crossing_region = [(cross_x1, cross_y1), (cross_x2, cross_y1), (cross_x2, cross_y2), (cross_x1, cross_y2)]
    
    # Preview
    preview = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.polylines(preview, [np.array(lane_region, dtype=np.int32)], True, (0, 255, 255), 2)
    cv2.polylines(preview, [np.array(crossing_region, dtype=np.int32)], True, (255, 255, 0), 2)
    cv2.putText(preview, "FREE LEFT LANE", (lane_x1, lane_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(preview, "PEDESTRIAN CROSSING", (cross_x1, cross_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    st.image(preview, caption="Region Preview", use_container_width=True)
    
    return lane_region, crossing_region


def render_video_source():
    """Render video source selection"""
    st.subheader("📹 Video Source")
    
    source_type = st.radio(
        "Source Type",
        ["📷 Live Camera", "🎬 Video File Upload"],
        horizontal=True,
        key="source_type"
    )
    
    if source_type == "📷 Live Camera":
        camera_id = st.selectbox("Camera Device", ["Webcam (0)", "External Camera (1)", "USB Camera (2)"], index=0)
        camera_index = int(camera_id.split("(")[1].split(")")[0])
        return {'type': 'camera', 'source': camera_index}
    else:
        uploaded = st.file_uploader(
            "Upload Video File", 
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload pre-recorded traffic video for testing"
        )
        if uploaded:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"uploaded_video_{int(time.time())}.mp4")
            with open(temp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            st.session_state.app_state.video_file_path = temp_path
            
            # Display video info
            cap = cv2.VideoCapture(temp_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resolution", f"{width}x{height}")
            with col2:
                st.metric("FPS", f"{fps:.1f}")
            with col3:
                st.metric("Duration", f"{duration:.1f}s")
            
            return {'type': 'video', 'source': temp_path, 'name': uploaded.name}
        return None


def render_signal_status():
    """Render current signal status"""
    app = st.session_state.app_state
    state = app.controller.get_state()
    
    st.subheader("🚥 Current Signal Status")
    
    phase = state['phase']
    
    if "FREE" in phase:
        st.markdown(f"""
        <div class="signal-free">
            <h2>{phase}</h2>
            <p>Yield to oncoming traffic | Scan for gaps</p>
            <p style="font-size: 0.9rem;">⚠️ Proceed with caution</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PROTECTED" in phase:
        st.markdown(f"""
        <div class="signal-protected">
            <h2>{phase}</h2>
            <p>Green arrow - Safe to turn | Oncoming traffic stopped</p>
            <p style="font-size: 0.9rem;">✅ Clear intersection</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PEDESTRIAN" in phase:
        st.markdown(f"""
        <div class="signal-pedestrian">
            <h2>{phase}</h2>
            <p>Stop for pedestrians | All vehicles must yield</p>
            <p style="font-size: 0.9rem;">🚶 Pedestrians crossing</p>
        </div>
        """, unsafe_allow_html=True)
    elif "EMERGENCY" in phase:
        st.markdown(f"""
        <div class="signal-emergency">
            <h2>{phase}</h2>
            <p>EMERGENCY STOP | Conflict detected</p>
            <p style="font-size: 0.9rem;">⚠️ ALL VEHICLES STOP</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🚗 Blocking", state['blocking_vehicles'], delta=None)
    with col2:
        st.metric("🚶 Pedestrians", state.get('pedestrians_waiting', 0))
    with col3:
        st.metric("⏱️ Phase Duration", f"{state['phase_duration']:.0f}s")
    with col4:
        risk = state['risk_level']
        risk_color = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
        st.metric("📊 Risk Level", f"{risk_color} {risk}")
    with col5:
        if state.get('is_peak_hour', False):
            st.metric("⏰ Peak Hour", "ACTIVE", delta="Enhanced sensitivity")
        else:
            st.metric("⏰ Peak Hour", "NORMAL")
    
    # Cooldown progress
    if state['cooldown_remaining'] > 0:
        st.progress(min(state['cooldown_remaining'] / 60, 1.0))
        st.caption(f"Cooldown: {state['cooldown_remaining']:.0f}s remaining")


def render_realtime_feed():
    """Render real-time detection feed"""
    app = st.session_state.app_state
    
    if not app.is_running or not app.video_source:
        st.info("📹 No active source. Configure settings and click START.")
        return
    
    # Read frame
    ret, frame = app.video_source.read()
    if not ret:
        if app.video_source and hasattr(app.video_source, 'get'):
            # Try to reset video for replay
            if hasattr(app.video_source, 'set'):
                app.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = app.video_source.read()
        if not ret:
            st.warning("End of video reached or no frame available")
            return
    
    # Detect
    detections, annotated = app.detector.detect(frame)
    
    # Get lane vehicles and pedestrians
    lane_vehicles = [d for d in detections if d.type == DetectionType.VEHICLE and d.in_lane]
    pedestrians = [d for d in detections if d.type == DetectionType.PEDESTRIAN and d.at_crossing]
    
    # Update controller
    phase = app.controller.update_detections(lane_vehicles, pedestrians)
    
    # Update stats history
    stats = app.detector.get_stats()
    app.stats_history.append({
        'timestamp': time.time(),
        'blocking': len(lane_vehicles),
        'pedestrians': len(pedestrians),
        'fps': stats['fps']
    })
    if len(app.stats_history) > 100:
        app.stats_history.pop(0)
    
    # Display frame
    st.image(annotated, channels="BGR", use_container_width=True, caption="Live Detection Feed")
    
    # Detection info panels
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚗 Free-Left Lane")
        if lane_vehicles:
            st.warning(f"⚠️ {len(lane_vehicles)} vehicle(s) in free-left lane")
            for v in lane_vehicles[:3]:
                blocking_status = "🔴 BLOCKING" if v.stationary_time > 3 else "🟡 In lane"
                st.write(f"• {v.class_name} - {blocking_status} ({v.stationary_time:.1f}s)")
        else:
            st.success("✅ Lane clear")
    
    with col2:
        st.markdown("### 🚶 Pedestrian Crossing")
        if pedestrians:
            if app.detector.get_crossing_region():
                st.info(f"🚶 {len(pedestrians)} pedestrian(s) at crossing")
                for p in pedestrians[:3]:
                    st.write(f"• Waiting to cross - {p.confidence:.2f} conf")
            else:
                st.warning("⚠️ No crossing detected - pedestrians may be at risk")
        else:
            st.success("✅ No pedestrians")
    
    with col3:
        st.markdown("### 📊 Detection Stats")
        stats = app.detector.get_stats()
        st.metric("FPS", f"{stats['fps']:.1f}")
        st.metric("Tracked Vehicles", stats['tracked_vehicles'])
        st.metric("Tracked Pedestrians", stats['tracked_pedestrians'])
        
        if stats.get('crossing_detected', False):
            st.markdown('<span class="status-badge status-active">✓ Crossing Detected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">⚠ Scanning for crossing...</span>', unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(0.03)
    st.rerun()


def render_analytics():
    """Render analytics dashboard"""
    app = st.session_state.app_state
    
    st.subheader("📊 System Analytics")
    
    # Real-time chart
    if app.stats_history:
        df = pd.DataFrame(app.stats_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['blocking'], name='Blocking Vehicles', mode='lines', line=dict(color='red')))
        fig.add_trace(go.Scatter(y=df['pedestrians'], name='Pedestrians', mode='lines', line=dict(color='blue')))
        fig.update_layout(title='Real-time Metrics', xaxis_title='Time', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    # Event distribution
    events = app.controller.get_events(100)
    if events:
        col1, col2 = st.columns(2)
        
        with col1:
            event_counts = {}
            for e in events:
                event_counts[e['event']] = event_counts.get(e['event'], 0) + 1
            
            if event_counts:
                df_events = pd.DataFrame(list(event_counts.items()), columns=['Event', 'Count'])
                fig = px.pie(df_events, values='Count', names='Event', title="Event Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Timeline
            df_timeline = pd.DataFrame(events)
            df_timeline['timestamp'] = pd.to_datetime(df_timeline['timestamp'])
            df_timeline['hour'] = df_timeline['timestamp'].dt.hour
            hourly = df_timeline['hour'].value_counts().sort_index()
            
            if not hourly.empty:
                fig = go.Figure(data=[go.Bar(x=hourly.index.astype(str), y=hourly.values)])
                fig.update_layout(title="Events by Hour", xaxis_title="Hour", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
    
    # Event log
    with st.expander("📋 Event Log"):
        for event in reversed(events[-20:]):
            timestamp = event['timestamp'][:19]
            event_type = event['event']
            data = event.get('data', {})
            
            if event_type == 'PROTECTED_TRIGGERED':
                st.warning(f"**{timestamp}** - 🟢 PROTECTED LEFT | {data.get('violations', 0)} vehicles blocking")
            elif event_type == 'PEDESTRIAN_MODE':
                st.info(f"**{timestamp}** - 🚶 PEDESTRIAN MODE | {data.get('count', 0)} pedestrians waiting")
            elif event_type == 'EMERGENCY_STOP':
                st.error(f"**{timestamp}** - 🔴 EMERGENCY STOP | Conflict detected")
            else:
                st.write(f"**{timestamp}** - {event_type}")


def render_dataset_results():
    """Render dataset analysis results"""
    results = st.session_state.app_state.analysis_results
    if not results:
        return
    
    st.subheader("📊 Dataset Analysis Results")
    
    risk = results.get('risk_assessment', {})
    risk_level = risk.get('level', 'UNKNOWN')
    risk_class = f"risk-{risk_level.lower()}"
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h3>Risk Assessment: {risk_level} (Score: {risk.get('score', 0)}/100)</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Violations", results.get('violation_analysis', {}).get('total_violations', 0))
    with col2:
        st.metric("Risk Score", f"{risk.get('score', 0)}/100")
    with col3:
        peak = results.get('violation_analysis', {}).get('peak_hours', {})
        if peak:
            st.metric("Peak Hours", ', '.join([f"{h}:00" for h in peak.keys()]))
    with col4:
        st.metric("Recommendations", len(results.get('recommendations', [])))
    
    # Vehicle distribution
    vehicle_dist = results.get('violation_analysis', {}).get('vehicle_distribution', {})
    if vehicle_dist:
        st.markdown("### 🚗 Vehicle Type Distribution")
        df_vehicles = pd.DataFrame(list(vehicle_dist.items()), columns=['Vehicle Type', 'Count'])
        fig = px.pie(df_vehicles, values='Count', names='Vehicle Type', title='Violations by Vehicle Type')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    recs = results.get('recommendations', [])
    if recs:
        st.markdown("### 💡 Recommendations")
        for rec in recs[:5]:
            if "IMMEDIATE" in rec or "HIGH" in rec:
                st.error(rec)
            elif "Schedule" in rec or "Monitor" in rec:
                st.warning(rec)
            else:
                st.info(rec)


def render_download_report():
    """Render download report button"""
    results = st.session_state.app_state.analysis_results
    if not results:
        return
    
    st.markdown("---")
    st.subheader("📥 Download Analysis Report")
    
    if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Generating report..."):
            try:
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                story = []
                styles = getSampleStyleSheet()
                
                story.append(Spacer(1, 2*inch))
                story.append(Paragraph("Intelligent Free Left Turn Management", styles['Title']))
                story.append(Paragraph(f"Analysis Report: {st.session_state.app_state.intersection_name}", styles['Normal']))
                story.append(Spacer(1, 1*inch))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", styles['Normal']))
                story.append(Spacer(1, 0.5*inch))
                
                risk = results.get('risk_assessment', {})
                story.append(Paragraph(f"Risk Level: {risk.get('level', 'UNKNOWN')} (Score: {risk.get('score', 0)}/100)", styles['Normal']))
                story.append(Spacer(1, 0.5*inch))
                
                story.append(Paragraph("Recommendations:", styles['Heading2']))
                for rec in results.get('recommendations', [])[:5]:
                    story.append(Paragraph(f"• {rec}", styles['Normal']))
                
                doc.build(story)
                buffer.seek(0)
                
                b64 = base64.b64encode(buffer.getvalue()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="Free_Left_Turn_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf" style="background-color: #27ae60; color: white; padding: 0.75rem 1.5rem; border-radius: 5px; text-decoration: none; display: inline-block;">📥 Click to download report</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Report generated!")
                
            except Exception as e:
                st.error(f"Error: {e}")


def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.header("🎮 System Controls")
        
        mode = st.radio(
            "Select Mode",
            ["📊 Dataset Analysis", "📹 Real-time Monitoring"],
            help="Choose between analyzing uploaded datasets or real-time monitoring"
        )
        
        st.divider()
        
        if mode == "📹 Real-time Monitoring":
            # Detection Settings
            st.subheader("⚙️ Detection Settings")
            
            # Auto zebra crossing detection
            auto_detect = st.checkbox(
                "🤖 Auto-Detect Zebra Crossing", 
                value=st.session_state.app_state.auto_detect_crossing,
                help="Automatically detect pedestrian crossings from camera feed"
            )
            st.session_state.app_state.auto_detect_crossing = auto_detect
            
            confidence = st.slider("Detection Confidence", 0.3, 0.8, 0.4, 0.05,
                                  help="Higher = fewer false positives, Lower = more detections")
            
            # Manual region config (if auto-detect is off)
            lane_region = None
            crossing_region = None
            
            if not auto_detect:
                lane_region, crossing_region = render_manual_region_config()
            
            st.divider()
            
            # Source selection
            source = render_video_source()
            
            st.divider()
            
            # Start/Stop buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ START", type="primary", use_container_width=True):
                    if source:
                        app = st.session_state.app_state
                        
                        # Create detector
                        app.detector = UnifiedDetector(
                            lane_region=lane_region if not auto_detect else None,
                            auto_detect_crossing=auto_detect,
                            conf_threshold=confidence
                        )
                        
                        # Create video source
                        if source['type'] == 'camera':
                            app.video_source = cv2.VideoCapture(source['source'])
                        else:
                            app.video_source = cv2.VideoCapture(source['source'])
                        
                        if app.video_source.isOpened():
                            app.is_running = True
                            st.success(f"✅ Started! Mode: {'Auto Zebra Detection' if auto_detect else 'Manual Regions'}")
                            st.rerun()
                        else:
                            st.error("Cannot open source")
                    else:
                        st.warning("Please select a source first")
            
            with col2:
                if st.button("⏹️ STOP", use_container_width=True):
                    app = st.session_state.app_state
                    app.is_running = False
                    if app.video_source:
                        app.video_source.release()
                    app.detector = None
                    app.video_source = None
                    st.info("Stopped")
                    st.rerun()
            
            st.divider()
            
            # Manual override
            st.subheader("🔧 Manual Override")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔒 Protected Left", use_container_width=True):
                    app = st.session_state.app_state
                    app.controller.manual_protect()
                    st.success("Protected left activated")
            with col2:
                if st.button("🚶 Pedestrian Mode", use_container_width=True):
                    app = st.session_state.app_state
                    app.controller.manual_pedestrian()
                    st.success("Pedestrian mode activated")
            
            if st.button("🔄 Reset to Auto", use_container_width=True):
                app = st.session_state.app_state
                app.controller.manual_reset()
                st.info("Auto mode restored")
        
        else:
            # Dataset analysis mode
            st.subheader("📄 Upload Dataset")
            
            uploaded = st.file_uploader("Upload Traffic Data PDF", type=['pdf'],
                                        help="Upload traffic department PDF report")
            
            if uploaded:
                st.info(f"📁 File: {uploaded.name} ({uploaded.size / 1024:.1f} KB)")
                
                if st.button("🔍 Analyze Dataset", type="primary", use_container_width=True):
                    with st.spinner("Analyzing dataset..."):
                        try:
                            temp_dir = tempfile.gettempdir()
                            temp_path = os.path.join(temp_dir, uploaded.name)
                            with open(temp_path, 'wb') as f:
                                f.write(uploaded.getbuffer())
                            
                            parser = PDFParser()
                            data = parser.parse(temp_path)
                            
                            intersection_name = data.get('intersection_info', {}).get('name', 'Banashankari Junction')
                            st.session_state.app_state.intersection_name = intersection_name
                            
                            analyzer = DataAnalyzer(data)
                            analysis = analyzer.analyze()
                            
                            st.session_state.app_state.analysis_results = analysis
                            st.session_state.analysis_complete = True
                            st.session_state.app_state.controller.integrate_dataset(analysis)
                            
                            st.success("✅ Dataset analyzed successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        return mode


def main():
    """Main application entry point"""
    init_session_state()
    render_header()
    
    mode = render_sidebar()
    
    if mode == "📹 Real-time Monitoring":
        # Two-column layout for real-time monitoring
        col1, col2 = st.columns([3, 2])
        
        with col1:
            render_realtime_feed()
        
        with col2:
            render_signal_status()
            render_analytics()
    
    else:
        # Dataset analysis mode
        if st.session_state.analysis_complete and st.session_state.app_state.analysis_results:
            render_dataset_results()
            render_download_report()
        else:
            st.info("📄 Upload a PDF dataset from the sidebar to begin analysis")
        
        render_analytics()
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🕒 System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        status = "🟢 Active" if st.session_state.app_state.is_running else "⚪ Idle"
        st.caption(f"System Status: {status}")
    with col3:
        st.caption("🚦 Intelligent Free Left Turn Management v2.0")


if __name__ == "__main__":
    main()