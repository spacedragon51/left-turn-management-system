#!/usr/bin/env python3
"""
Intelligent Free Left Turn Management System
Complete Application with Professional UI for Hackathon Evaluation
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
from collections import deque

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Intelligent Free Left Turn Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Professional Styling
# ============================================================================
st.markdown("""
<style>
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.9;
    }
    
    /* Signal Cards */
    .signal-card {
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .signal-card:hover {
        transform: translateY(-5px);
    }
    .signal-free { background: linear-gradient(135deg, #ffa500, #ff8c00); color: white; }
    .signal-protected { background: linear-gradient(135deg, #00cc66, #00994d); color: white; animation: pulse 2s infinite; }
    .signal-pedestrian { background: linear-gradient(135deg, #ff4b4b, #cc0000); color: white; animation: blink 1s infinite; }
    .signal-emergency { background: linear-gradient(135deg, #8b0000, #4a0000); color: white; animation: blink 0.5s infinite; }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.95; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e3c72;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk Level Badges */
    .risk-high { 
        background: linear-gradient(135deg, #ff4b4b, #cc0000);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium { 
        background: linear-gradient(135deg, #ffa500, #ff8c00);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low { 
        background: linear-gradient(135deg, #00cc66, #00994d);
        color: white;
        padding: 0.5rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .status-active { background: #00cc66; color: white; }
    .status-warning { background: #ffa500; color: white; }
    .status-danger { background: #ff4b4b; color: white; }
    .status-info { background: #17a2b8; color: white; }
    
    /* Region Preview */
    .region-preview {
        background: #1e1e1e;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    
    /* Sidebar Styling */
    .sidebar-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORTS
# ============================================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import UnifiedDetector, DetectionType
from core.signal_controller import EnhancedSignalController, SignalPhase
from core.zebra_detector import EnhancedZebraDetector, ZebraCrossingTracker
from core.pedestrian_tracker import PedestrianTracker
from core.emergency_detector import EmergencyVehicleDetector
from core.fallback_controller import FallbackController
from core.pdf_parser import PDFParser
from core.data_analyzer import DataAnalyzer


# ============================================================================
# APP STATE MANAGEMENT
# ============================================================================
class AppState:
    """Application state management"""
    def __init__(self):
        self.detector = None
        self.controller = EnhancedSignalController()
        self.video_source = None
        self.pedestrian_camera = None
        self.is_running = False
        self.analysis_results = None
        self.intersection_name = "Banashankari Junction"
        self.lane_region = None
        self.crossing_region = None
        self.pedestrian_region = None
        self.auto_detect_crossing = True
        self.enhance_visibility = True
        self.debug_mode = False
        self.dual_camera_mode = False
        self.pedestrian_tracker_enabled = False
        self.crossing_detected = False
        self.stats_history = []
        self.video_file_path = None
        self.ped_video_path = None
        self.frame_width = 1280
        self.frame_height = 720
        self.pedestrian_stats = {}
        self.emergency_detector = None
        self.fallback_controller = None
        self.health_check_counter = 0


def init_session_state():
    """Initialize session state"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'lane_region_set' not in st.session_state:
        st.session_state.lane_region_set = False
    if 'crossing_region_set' not in st.session_state:
        st.session_state.crossing_region_set = False


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    """Render main header with branding"""
    st.markdown("""
    <div class="main-header">
        <h1>🚦 Intelligent Free Left Turn Management System</h1>
        <p>AI-Powered Traffic Management | Real-time Detection | Pedestrian Safety | Emergency Priority</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem;">⚡ Weighted Occupancy | 🚶 Pedestrian Tracking | 🚨 Emergency Override | 🔄 Fallback Safety</p>
    </div>
    """, unsafe_allow_html=True)


def render_signal_status():
    """Render current signal status with animation"""
    app = st.session_state.app_state
    state = app.controller.get_state()
    
    st.markdown("### 🚥 Current Signal Status")
    
    phase = state['phase']
    
    if "FREE" in phase:
        st.markdown(f"""
        <div class="signal-card signal-free">
            <h2 style="margin: 0;">{phase}</h2>
            <p style="margin: 0.5rem 0 0 0;">Yield to oncoming traffic | Scan for safe gaps</p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">⚠️ Driver must judge gap</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PROTECTED" in phase:
        st.markdown(f"""
        <div class="signal-card signal-protected">
            <h2 style="margin: 0;">{phase}</h2>
            <p style="margin: 0.5rem 0 0 0;">Green arrow - Safe to turn | Oncoming traffic stopped</p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">✅ System guaranteed safe turn</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PEDESTRIAN" in phase:
        st.markdown(f"""
        <div class="signal-card signal-pedestrian">
            <h2 style="margin: 0;">{phase}</h2>
            <p style="margin: 0.5rem 0 0 0;">🚶 Pedestrians crossing - All vehicles STOP</p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">🚸 Wait for pedestrians to clear</p>
        </div>
        """, unsafe_allow_html=True)
    elif "EMERGENCY" in phase:
        st.markdown(f"""
        <div class="signal-card signal-emergency">
            <h2 style="margin: 0;">{phase}</h2>
            <p style="margin: 0.5rem 0 0 0;">🚨 Emergency vehicle detected - All vehicles STOP</p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">🚑 Give way to emergency vehicles</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">🚗 Blocking</div>
        </div>
        """.format(state['blocking_vehicles']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">🚶 Pedestrians</div>
        </div>
        """.format(state.get('pedestrians_waiting', 0)), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.0f}s</div>
            <div class="metric-label">⏱️ Duration</div>
        </div>
        """.format(state['phase_duration']), unsafe_allow_html=True)
    
    with col4:
        risk = state['risk_level']
        risk_icon = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">📊 Risk</div>
        </div>
        """.format(f"{risk_icon} {risk}"), unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">⚖️ Weighted Occ</div>
        </div>
        """.format(state.get('weighted_threshold', 2.5)), unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">🔄 Triggers</div>
        </div>
        """.format(state.get('protected_triggers', 0)), unsafe_allow_html=True)
    
    # Status Messages
    if state.get('emergency_mode', False):
        st.error(f"🚨 EMERGENCY VEHICLE: {state.get('emergency_type', 'Unknown')} - All vehicles must yield!")
    
    if state.get('pedestrian_mode_active', False):
        st.info(f"🚶 Pedestrian Mode Active - {state.get('pedestrians_waiting', 0)} pedestrians waiting to cross")
    
    if state.get('cooldown_remaining', 0) > 0:
        st.progress(min(state['cooldown_remaining'] / 30, 1.0))
        st.caption(f"⏱️ Cooldown: {state['cooldown_remaining']:.0f}s remaining before next intervention")


def render_realtime_feed():
    """Render real-time detection feed with dual camera support"""
    app = st.session_state.app_state
    
    if not app.is_running:
        st.info("📹 No active source. Configure settings in sidebar and click START.")
        return
    
    # Read main camera frame
    if app.video_source is None:
        st.warning("Main camera source not available")
        return
        
    ret, frame = app.video_source.read()
    if not ret:
        if hasattr(app.video_source, 'set'):
            app.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = app.video_source.read()
        if not ret:
            st.warning("End of main video or no frame available")
            return
    
    # Read pedestrian camera frame
    ped_frame = None
    ped_camera_active = False
    
    if app.dual_camera_mode and app.pedestrian_camera is not None:
        ret_ped, ped_frame = app.pedestrian_camera.read()
        if ret_ped:
            ped_camera_active = True
        elif hasattr(app.pedestrian_camera, 'set'):
            app.pedestrian_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ped, ped_frame = app.pedestrian_camera.read()
            if ret_ped:
                ped_camera_active = True
    
    # Apply manual regions
    if not app.auto_detect_crossing:
        if app.lane_region and app.detector:
            app.detector.lane_region = app.lane_region
        if app.crossing_region and app.detector:
            app.detector.set_manual_crossing(app.crossing_region)
    
    # Main detection
    detections, annotated = app.detector.detect(frame)
    
    # Get lane vehicles and pedestrians
    lane_vehicles = [d for d in detections if d.type in [DetectionType.VEHICLE, DetectionType.MOTORCYCLE] and d.in_lane]
    pedestrians_main = [d for d in detections if d.type == DetectionType.PEDESTRIAN and d.at_crossing]
    
    # Pedestrian camera detection
    ped_annotated = None
    pedestrian_stats = {}
    
    if ped_camera_active and ped_frame is not None:
        if app.pedestrian_tracker_enabled and app.detector.pedestrian_tracker:
            ped_detections, _ = app.detector.detect(ped_frame)
            tracked_peds, pedestrian_stats = app.detector.pedestrian_tracker.update(ped_detections)
            ped_annotated = app.detector.pedestrian_tracker.draw(ped_frame)
        else:
            ped_annotated = ped_frame.copy()
            cv2.putText(ped_annotated, "PEDESTRIAN CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Calculate metrics
    weighted_occupancy = app.detector.get_weighted_occupancy(lane_vehicles)
    spillover = app.detector.detect_spillover(lane_vehicles)
    
    # Emergency detection
    emergency_status = {'emergency_detected': False}
    if app.emergency_detector:
        emergency_status = app.emergency_detector.update(frame)
    
    # Update controller
    phase = app.controller.update_detections(
        lane_vehicles, pedestrians_main,
        detector=app.detector,
        oncoming_count=0,
        weighted_occupancy=weighted_occupancy,
        spillover=spillover,
        emergency=emergency_status.get('emergency_detected', False),
        emergency_type=emergency_status.get('emergency_type')
    )
    
    # Health check
    if app.fallback_controller:
        app.health_check_counter += 1
        if app.health_check_counter % 30 == 0:
            app.fallback_controller.check_health(
                detection_available=len(detections) > 0,
                frame_rate=app.detector.fps if app.detector else 0
            )
    
    # Update stats
    stats = app.detector.get_stats()
    app.stats_history.append({
        'timestamp': time.time(),
        'blocking': len(lane_vehicles),
        'pedestrians': pedestrian_stats.get('waiting_count', len(pedestrians_main)),
        'weighted_occupancy': weighted_occupancy,
        'fps': stats['fps']
    })
    if len(app.stats_history) > 100:
        app.stats_history.pop(0)
    
    app.pedestrian_stats = pedestrian_stats
    
    # Display feeds
    has_valid_ped_feed = ped_camera_active and ped_annotated is not None
    
    if app.dual_camera_mode and has_valid_ped_feed:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### 🚗 Main Camera (Traffic View)")
            st.image(annotated, channels="BGR", use_container_width=True)
        with col2:
            st.markdown("#### 🚶 Pedestrian Camera")
            st.image(ped_annotated, channels="BGR", use_container_width=True) # pyright: ignore[reportArgumentType]
    else:
        st.image(annotated, channels="BGR", use_container_width=True, caption="Live Detection Feed")
        if app.dual_camera_mode and not ped_camera_active:
            st.info("💡 Dual camera mode enabled but pedestrian camera not available. Using single camera mode.")
    
    # Metrics Panels
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚗 Free-Left Lane")
        if lane_vehicles:
            st.warning(f"⚠️ {len(lane_vehicles)} vehicle(s) in lane")
            for v in lane_vehicles[:3]:
                icon = "🏍️" if v.type == DetectionType.MOTORCYCLE else "🚗"
                blocking = "🔴 BLOCKING" if v.is_stationary and v.stationary_time > 3 else "🟡 In lane"
                weight = f" [W:{v.weight:.1f}]" if v.weight > 1.0 else ""
                st.write(f"{icon} {v.class_name}{weight} - {blocking} ({v.stationary_time:.1f}s)")
            st.progress(min(weighted_occupancy / 5, 1.0))
            st.caption(f"Weighted Occupancy: {weighted_occupancy:.1f}")
            if spillover:
                st.error("⚠️ SPILLOVER DETECTED - Queue blocking straight lane")
        else:
            st.success("✅ Lane clear")
    
    with col2:
        st.markdown("### 🚶 Pedestrian Status")
        if pedestrian_stats:
            st.metric("Waiting", pedestrian_stats.get('waiting_count', 0))
            st.metric("Crossing", pedestrian_stats.get('crossing_count', 0))
            st.metric("Total Crossed", pedestrian_stats.get('total_crossed', 0))
            st.metric("Avg Wait", f"{pedestrian_stats.get('avg_wait_time', 0):.1f}s")
        elif pedestrians_main:
            st.info(f"🚶 {len(pedestrians_main)} pedestrian(s) at crossing")
        else:
            if app.crossing_region or app.detector.get_crossing_region():
                st.success("✅ No pedestrians")
            else:
                st.warning("⚠️ No crossing region configured")
    
    with col3:
        st.markdown("### 📊 System Stats")
        st.metric("FPS", f"{stats['fps']:.1f}")
        st.metric("Vehicles", stats['tracked_vehicles'])
        st.metric("Motorcycles", stats.get('tracked_motorcycles', 0))
        st.metric("Pedestrians", stats['tracked_pedestrians'])
        st.metric("Queue Length", f"{stats.get('queue_length', 0)}px")
        st.metric("Occlusion Factor", f"{stats.get('occlusion_factor', 0):.2f}")
        
        if emergency_status.get('emergency_detected', False):
            st.error(f"🚨 EMERGENCY: {emergency_status.get('emergency_type', 'Unknown')}")
        
        if app.fallback_controller:
            fb_status = app.fallback_controller.get_status()
            if fb_status['is_active']:
                st.warning(f"⚠️ FALLBACK MODE: {fb_status['mode']}")
    
    time.sleep(0.03)
    st.rerun()


def render_analytics():
    """Render analytics dashboard"""
    app = st.session_state.app_state
    
    st.markdown("### 📊 System Analytics")
    
    if app.stats_history:
        df = pd.DataFrame(app.stats_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['blocking'], name='Blocking Vehicles', mode='lines', line=dict(color='#ff4b4b', width=2)))
        fig.add_trace(go.Scatter(y=df['weighted_occupancy'], name='Weighted Occupancy', mode='lines', line=dict(color='#ffa500', width=2, dash='dash')))
        fig.add_trace(go.Scatter(y=df['pedestrians'], name='Pedestrians', mode='lines', line=dict(color='#17a2b8', width=2)))
        fig.update_layout(
            title='Real-time Metrics',
            xaxis_title='Time (frames)',
            yaxis_title='Count',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Events Log
    events = app.controller.get_events(100)
    if events:
        with st.expander("📋 Event Log", expanded=False):
            for event in reversed(events[-15:]):
                ts = time.strftime('%H:%M:%S', time.localtime(event['timestamp'])) if isinstance(event['timestamp'], (int, float)) else event['timestamp'][:19]
                if event['event'] == 'PROTECTED_TRIGGERED':
                    st.warning(f"**{ts}** - 🟢 PROTECTED LEFT | {event['data'].get('reason', '')}")
                elif event['event'] == 'PEDESTRIAN_MODE':
                    st.info(f"**{ts}** - 🚶 PEDESTRIAN MODE | {event['data'].get('count', 0)} waiting")
                elif event['event'] == 'EMERGENCY_MODE':
                    st.error(f"**{ts}** - 🚨 EMERGENCY MODE | {event['data'].get('type', 'Unknown')}")
                elif event['event'] == 'SPILLOVER_DETECTED':
                    st.warning(f"**{ts}** - ⚠️ SPILLOVER | {event['data'].get('queue', 0)} vehicles")
                else:
                    st.write(f"**{ts}** - {event['event']}")


def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.markdown("## 🎮 System Controls")
        
        mode = st.radio(
            "Select Mode",
            ["📊 Dataset Analysis", "📹 Real-time Monitoring"],
            help="Choose between analyzing uploaded datasets or real-time monitoring"
        )
        
        st.divider()
        
        if mode == "📹 Real-time Monitoring":
            st.markdown("### ⚙️ Detection Settings")
            
            auto_detect = st.checkbox(
                "🤖 Auto-Detect Zebra Crossing", 
                value=st.session_state.app_state.auto_detect_crossing,
                help="Automatically detect pedestrian crossings from camera feed"
            )
            st.session_state.app_state.auto_detect_crossing = auto_detect
            
            enhance_vis = st.checkbox(
                "✨ Enhance Visibility", 
                value=st.session_state.app_state.enhance_visibility,
                help="Apply image enhancement for poor lighting conditions"
            )
            st.session_state.app_state.enhance_visibility = enhance_vis
            
            debug_mode = st.checkbox(
                "🐛 Debug Mode", 
                value=st.session_state.app_state.debug_mode,
                help="Show debug information and visualizations"
            )
            st.session_state.app_state.debug_mode = debug_mode
            
            confidence = st.slider("Detection Confidence", 0.3, 0.8, 0.4, 0.05)
            
            st.divider()
            
            # Camera Configuration
            st.markdown("### 📹 Camera Configuration")
            
            camera_layout = st.radio(
                "Camera Setup",
                ["Single Camera", "Dual Camera (Main + Pedestrian)"],
                help="Choose camera setup for monitoring"
            )
            st.session_state.app_state.dual_camera_mode = (camera_layout == "Dual Camera (Main + Pedestrian)")
            
            # Main Camera Source
            st.markdown("#### 📷 Main Camera")
            source_type = st.radio(
                "Main Source Type",
                ["Live Camera", "Video File"],
                horizontal=True,
                key="main_source_type"
            )
            
            main_source = None
            if source_type == "Live Camera":
                camera_id = st.selectbox("Camera Device", ["Webcam (0)", "External (1)"], index=0)
                camera_index = int(camera_id.split("(")[1].split(")")[0])
                main_source = {'type': 'camera', 'source': camera_index, 'name': f'Camera {camera_index}'}
            else:
                uploaded = st.file_uploader("Upload Main Video", type=['mp4', 'avi', 'mov'], key="main_video")
                if uploaded:
                    temp_path = os.path.join(tempfile.gettempdir(), f"main_video_{int(time.time())}.mp4")
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded.getbuffer())
                    main_source = {'type': 'video', 'source': temp_path, 'name': uploaded.name}
                    st.success(f"✅ Loaded: {uploaded.name}")
            
            # Pedestrian Camera Source
            ped_source = None
            if st.session_state.app_state.dual_camera_mode:
                st.markdown("#### 🚶 Pedestrian Camera")
                ped_source_option = st.radio(
                    "Pedestrian Source",
                    ["Same as Main", "Separate Camera/Video"],
                    horizontal=True,
                    key="ped_source_option"
                )
                
                if ped_source_option == "Separate Camera/Video":
                    ped_type = st.radio("Pedestrian Source Type", ["Live Camera", "Video File"], horizontal=True, key="ped_type")
                    if ped_type == "Live Camera":
                        ped_cam = st.selectbox("Ped Camera", ["Webcam (0)", "External (1)"], index=1, key="ped_cam")
                        ped_index = int(ped_cam.split("(")[1].split(")")[0])
                        ped_source = {'type': 'camera', 'source': ped_index, 'name': f'Camera {ped_index}'}
                    else:
                        ped_upload = st.file_uploader("Upload Pedestrian Video", type=['mp4', 'avi', 'mov'], key="ped_video")
                        if ped_upload:
                            ped_temp = os.path.join(tempfile.gettempdir(), f"ped_video_{int(time.time())}.mp4")
                            with open(ped_temp, 'wb') as f:
                                f.write(ped_upload.getbuffer())
                            ped_source = {'type': 'video', 'source': ped_temp, 'name': ped_upload.name}
                            st.success(f"✅ Loaded: {ped_upload.name}")
                else:
                    ped_source = main_source
            
            st.divider()
            
            # Manual Region Configuration
            if not auto_detect:
                st.markdown("### 📍 Manual Region Setup")
                st.warning("Auto-detection is OFF - Define regions manually")
                
                width = st.session_state.app_state.frame_width
                height = st.session_state.app_state.frame_height
                
                st.markdown("#### 🟡 Free-Left Lane")
                lane_x1 = st.number_input("Lane Left X", 0, width, width//10, key="lane_x1")
                lane_y1 = st.number_input("Lane Top Y", 0, height, height-150, key="lane_y1")
                lane_x2 = st.number_input("Lane Right X", 0, width, width//3, key="lane_x2")
                lane_y2 = st.number_input("Lane Bottom Y", 0, height, height-120, key="lane_y2")
                lane_region = [(lane_x1, lane_y1), (lane_x2, lane_y1), (lane_x2, lane_y2), (lane_x1, lane_y2)]
                
                st.markdown("#### 🟢 Pedestrian Crossing")
                cross_x1 = st.number_input("Cross Left X", 0, width, width//3, key="cross_x1")
                cross_y1 = st.number_input("Cross Top Y", 0, height, height//2-50, key="cross_y1")
                cross_x2 = st.number_input("Cross Right X", 0, width, 2*width//3, key="cross_x2")
                cross_y2 = st.number_input("Cross Bottom Y", 0, height, height//2+50, key="cross_y2")
                crossing_region = [(cross_x1, cross_y1), (cross_x2, cross_y1), (cross_x2, cross_y2), (cross_x1, cross_y2)]
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Apply Lane", use_container_width=True):
                        st.session_state.app_state.lane_region = lane_region
                        st.session_state.lane_region_set = True
                        st.success("Lane region applied")
                with col2:
                    if st.button("✅ Apply Crossing", use_container_width=True):
                        st.session_state.app_state.crossing_region = crossing_region
                        st.session_state.crossing_region_set = True
                        st.success("Crossing region applied")
            
            st.divider()
            
            # Emergency Override
            st.markdown("### 🚨 Emergency Override")
            if st.button("🚨 Activate Emergency Mode", type="primary", use_container_width=True):
                if st.session_state.app_state.emergency_detector:
                    st.session_state.app_state.emergency_detector.manual_override(True, "Manual")
                st.success("Emergency mode activated!")
            
            st.divider()
            
            # Start/Stop
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ START", type="primary", use_container_width=True):
                    if main_source:
                        app = st.session_state.app_state
                        
                        app.detector = UnifiedDetector(
                            auto_detect_crossing=auto_detect,
                            conf_threshold=confidence,
                            enhance_visibility=enhance_vis,
                            debug_mode=debug_mode
                        )
                        
                        if not auto_detect:
                            if app.lane_region:
                                app.detector.lane_region = app.lane_region
                            if app.crossing_region:
                                app.detector.set_manual_crossing(app.crossing_region)
                        
                        app.emergency_detector = EmergencyVehicleDetector()
                        app.fallback_controller = FallbackController()
                        
                        if main_source['type'] == 'camera':
                            app.video_source = cv2.VideoCapture(main_source['source'])
                        else:
                            app.video_source = cv2.VideoCapture(main_source['source'])
                        
                        if ped_source and ped_source.get('type') == 'camera':
                            app.pedestrian_camera = cv2.VideoCapture(ped_source['source'])
                        elif ped_source and ped_source.get('type') == 'video':
                            app.pedestrian_camera = cv2.VideoCapture(ped_source['source'])
                        
                        if app.video_source.isOpened():
                            app.is_running = True
                            st.success("✅ System Started!")
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
                    if app.pedestrian_camera:
                        app.pedestrian_camera.release()
                    if app.emergency_detector:
                        app.emergency_detector.shutdown()
                    app.detector = None
                    app.video_source = None
                    app.pedestrian_camera = None
                    st.info("Stopped")
                    st.rerun()
            
            st.divider()
            
            # Manual Override
            st.markdown("### 🔧 Manual Override")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔒 Protected Left", use_container_width=True):
                    st.session_state.app_state.controller.manual_protect()
                    st.success("Protected left activated")
            with col2:
                if st.button("🚶 Pedestrian Mode", use_container_width=True):
                    st.session_state.app_state.controller.manual_pedestrian()
                    st.success("Pedestrian mode activated")
            
            if st.button("🔄 Reset to Auto", use_container_width=True):
                st.session_state.app_state.controller.manual_reset()
                if st.session_state.app_state.emergency_detector:
                    st.session_state.app_state.emergency_detector.reset()
                if st.session_state.app_state.fallback_controller:
                    st.session_state.app_state.fallback_controller.reset()
                st.info("Auto mode restored")
        
        else:
            # Dataset Analysis Mode
            st.markdown("### 📄 Upload Dataset")
            uploaded = st.file_uploader("Upload Traffic Data PDF", type=['pdf'])
            
            if uploaded:
                if st.button("🔍 Analyze Dataset", type="primary", use_container_width=True):
                    with st.spinner("Analyzing dataset..."):
                        try:
                            temp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
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
            
            if st.session_state.analysis_complete and st.session_state.app_state.analysis_results:
                results = st.session_state.app_state.analysis_results
                risk = results.get('risk_assessment', {})
                st.markdown(f"""
                <div class="risk-{risk.get('level', 'MEDIUM').lower()}">
                    <h4>Risk Assessment: {risk.get('level', 'UNKNOWN')} (Score: {risk.get('score', 0)}/100)</h4>
                </div>
                """, unsafe_allow_html=True)
                
                recs = results.get('recommendations', [])
                if recs:
                    with st.expander("💡 Recommendations"):
                        for rec in recs[:3]:
                            st.write(f"• {rec}")
        
        return mode


def render_dataset_results():
    """Render dataset analysis results"""
    results = st.session_state.app_state.analysis_results
    if not results:
        return
    
    st.markdown("### 📊 Dataset Analysis Results")
    
    risk = results.get('risk_assessment', {})
    risk_level = risk.get('level', 'UNKNOWN')
    
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
    
    recs = results.get('recommendations', [])
    if recs:
        st.markdown("### 💡 Recommendations")
        for rec in recs[:5]:
            st.info(rec)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
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
        if st.session_state.analysis_complete and st.session_state.app_state.analysis_results:
            render_dataset_results()
        else:
            st.info("📄 Upload a PDF dataset from the sidebar to begin analysis")
        render_analytics()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🚦 Intelligent Free Left Turn Management System v2.0 | AI-Powered Traffic Management</p>
        <p>⚡ Weighted Occupancy | 🚶 Pedestrian Tracking | 🚨 Emergency Priority | 🔄 Fallback Safety</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()