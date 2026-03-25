#!/usr/bin/env python3
"""
Intelligent Free Left Turn Management System
Main Streamlit Application - COMPLETE with Pedestrian Tracking Integration
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
    .signal-free { background-color: #ffa500; padding: 1rem; border-radius: 10px; text-align: center; color: white; }
    .signal-protected { background-color: #00cc66; padding: 1rem; border-radius: 10px; text-align: center; color: white; }
    .signal-pedestrian { background-color: #ff4b4b; padding: 1rem; border-radius: 10px; text-align: center; color: white; animation: pulse 1s infinite; }
    .signal-emergency { background-color: #8b0000; padding: 1rem; border-radius: 10px; text-align: center; color: white; animation: blink 0.5s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    @keyframes blink { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .risk-high { background-color: #ff4b4b; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; }
    .risk-medium { background-color: #ffa500; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; }
    .risk-low { background-color: #00cc66; color: white; padding: 0.5rem; border-radius: 5px; text-align: center; }
    .metric-card { background: #f0f2f6; padding: 1rem; border-radius: 10px; text-align: center; transition: transform 0.2s; }
    .metric-card:hover { transform: scale(1.02); }
    .region-preview { border: 2px solid #ddd; border-radius: 10px; padding: 10px; margin: 10px 0; background: #1e1e1e; }
    .pedestrian-stats { background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); padding: 0.5rem; border-radius: 10px; color: white; }
</style>
""", unsafe_allow_html=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import UnifiedDetector, DetectionType
from core.signal_controller import EnhancedSignalController, SignalPhase
from core.zebra_detector import EnhancedZebraDetector, ZebraCrossingTracker
from core.pedestrian_tracker import PedestrianTracker, DualCameraPedestrianManager
from core.pdf_parser import PDFParser
from core.data_analyzer import DataAnalyzer


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
    if 'pedestrian_tracker_active' not in st.session_state:
        st.session_state.pedestrian_tracker_active = False


def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🚦 Intelligent Free Left Turn Management System</h1>
        <p>AI-powered traffic management | Dual Camera | Pedestrian Tracking | Real-time Monitoring</p>
    </div>
    """, unsafe_allow_html=True)


def render_pedestrian_region_config():
    """Render pedestrian region configuration"""
    st.subheader("🚶 Pedestrian Region Configuration")
    st.info("Define the waiting and crossing areas for pedestrian tracking")
    
    width = st.session_state.app_state.frame_width
    height = st.session_state.app_state.frame_height
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Waiting Area**")
        wait_x1 = st.number_input("Wait Left X", 0, width, width // 3, key="wait_x1")
        wait_y1 = st.number_input("Wait Top Y", 0, height, height // 2 - 80, key="wait_y1")
        wait_x2 = st.number_input("Wait Right X", 0, width, 2 * width // 3, key="wait_x2")
        wait_y2 = st.number_input("Wait Bottom Y", 0, height, height // 2 - 20, key="wait_y2")
        waiting_region = [(wait_x1, wait_y1), (wait_x2, wait_y1), (wait_x2, wait_y2), (wait_x1, wait_y2)]
    
    with col2:
        st.markdown("**Crossing Area**")
        cross_x1 = st.number_input("Cross Left X", 0, width, width // 3, key="cross_x1")
        cross_y1 = st.number_input("Cross Top Y", 0, height, height // 2 - 50, key="cross_y1")
        cross_x2 = st.number_input("Cross Right X", 0, width, 2 * width // 3, key="cross_x2")
        cross_y2 = st.number_input("Cross Bottom Y", 0, height, height // 2 + 30, key="cross_y2")
        crossing_region = [(cross_x1, cross_y1), (cross_x2, cross_y1), (cross_x2, cross_y2), (cross_x1, cross_y2)]
    
    # Preview
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(preview, (0, height - 250), (width, height), (50, 50, 50), -1)
    
    # Draw waiting region (blue)
    wait_pts = np.array(waiting_region, dtype=np.int32)
    cv2.polylines(preview, [wait_pts], True, (255, 0, 0), 2)
    cv2.putText(preview, "WAITING AREA", (wait_x1, wait_y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw crossing region (green)
    cross_pts = np.array(crossing_region, dtype=np.int32)
    cv2.polylines(preview, [cross_pts], True, (0, 255, 0), 3)
    cv2.putText(preview, "CROSSING AREA", (cross_x1, cross_y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    st.image(preview, caption="Pedestrian Region Preview - Blue: Waiting, Green: Crossing", use_container_width=True)
    
    if st.button("✅ Apply Pedestrian Regions", key="apply_ped_regions"):
        st.session_state.app_state.pedestrian_region = {
            'waiting': waiting_region,
            'crossing': crossing_region
        }
        st.success("Pedestrian regions applied!")
    
    return waiting_region, crossing_region


def render_video_source():
    """Render video source selection"""
    st.subheader("📹 Video Source")
    
    source_type = st.radio(
        "Source Type",
        ["📷 Live Camera", "🎬 Video File Upload"],
        horizontal=True,
        key="source_type_main"
    )
    
    if source_type == "📷 Live Camera":
        camera_id = st.selectbox("Camera Device", ["Webcam (0)", "External (1)", "USB Camera (2)"], index=0)
        camera_index = int(camera_id.split("(")[1].split(")")[0])
        
        test_cap = cv2.VideoCapture(camera_index)
        if test_cap.isOpened():
            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.session_state.app_state.frame_width = width
            st.session_state.app_state.frame_height = height
            test_cap.release()
        
        return {'type': 'camera', 'source': camera_index}
    else:
        uploaded = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded:
            temp_path = os.path.join(tempfile.gettempdir(), f"uploaded_video_{int(time.time())}.mp4")
            with open(temp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            st.session_state.app_state.video_file_path = temp_path
            
            cap = cv2.VideoCapture(temp_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            st.session_state.app_state.frame_width = width
            st.session_state.app_state.frame_height = height
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


def render_pedestrian_camera():
    """Render pedestrian camera source selection"""
    st.subheader("🚶 Pedestrian Camera Source")
    
    source_type = st.radio(
        "Pedestrian Camera",
        ["📷 Same as Main Camera", "🎥 Separate Camera/Video"],
        horizontal=True,
        key="ped_source_type"
    )
    
    if source_type == "📷 Same as Main Camera":
        return {'type': 'same', 'source': None}
    else:
        ped_source = st.radio(
            "Pedestrian Source Type",
            ["Live Camera", "Video File Upload"],
            horizontal=True,
            key="ped_source_choice"
        )
        
        if ped_source == "Live Camera":
            camera_id = st.selectbox("Ped Camera Device", ["Webcam (0)", "External (1)", "USB Camera (2)"], index=1)
            camera_index = int(camera_id.split("(")[1].split(")")[0])
            return {'type': 'camera', 'source': camera_index}
        else:
            uploaded = st.file_uploader("Upload Pedestrian Video", type=['mp4', 'avi', 'mov', 'mkv'], key="ped_video")
            if uploaded:
                temp_path = os.path.join(tempfile.gettempdir(), f"ped_video_{int(time.time())}.mp4")
                with open(temp_path, 'wb') as f:
                    f.write(uploaded.getbuffer())
                st.session_state.app_state.ped_video_path = temp_path
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
            <p>🚦 Vehicles must yield to pedestrians</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PROTECTED" in phase:
        st.markdown(f"""
        <div class="signal-protected">
            <h2>{phase}</h2>
            <p>Green arrow - Safe to turn | Oncoming traffic stopped</p>
            <p>✅ Clear intersection</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PEDESTRIAN" in phase:
        st.markdown(f"""
        <div class="signal-pedestrian">
            <h2>{phase}</h2>
            <p>🚶 PEDESTRIAN CROSSING - All vehicles STOP</p>
            <p>Wait for pedestrians to cross safely</p>
        </div>
        """, unsafe_allow_html=True)
    elif "EMERGENCY" in phase:
        st.markdown(f"""
        <div class="signal-emergency">
            <h2>{phase}</h2>
            <p>🚨 EMERGENCY STOP | Conflict detected</p>
            <p>Immediate stop required</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🚗 Blocking", state['blocking_vehicles'])
    with col2:
        st.metric("🚶 Pedestrians", state.get('pedestrians_waiting', 0))
    with col3:
        st.metric("⏱️ Duration", f"{state['phase_duration']:.0f}s")
    with col4:
        risk = state['risk_level']
        risk_icon = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"
        st.metric("📊 Risk", f"{risk_icon} {risk}")
    with col5:
        threshold = state.get('current_threshold', 5)
        st.metric("🎯 Threshold", f"{threshold} vehicles")
    
    # Show cooldown progress
    if state['cooldown_remaining'] > 0:
        st.progress(min(state['cooldown_remaining'] / 30, 1.0))
        st.caption(f"Cooldown: {state['cooldown_remaining']:.0f}s remaining")
    
    if state.get('pedestrian_mode_active', False):
        st.info(f"🚶 Pedestrian Mode Active - {state.get('pedestrians_waiting', 0)} pedestrians waiting")


def render_realtime_feed():
    """Render real-time detection feed with pedestrian tracking"""
    app = st.session_state.app_state
    
    if not app.is_running or not app.video_source:
        st.info("📹 No active source. Configure settings and click START.")
        return
    
    # Read main camera frame
    ret, frame = app.video_source.read()
    if not ret:
        if hasattr(app.video_source, 'set'):
            app.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = app.video_source.read()
        if not ret:
            st.warning("End of video or no frame available")
            return
    
    # Read pedestrian camera frame if in dual camera mode
    ped_frame = None
    if app.dual_camera_mode and app.pedestrian_camera:
        ret_ped, ped_frame = app.pedestrian_camera.read()
        if not ret_ped and hasattr(app.pedestrian_camera, 'set'):
            app.pedestrian_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ped, ped_frame = app.pedestrian_camera.read()
    
    # Ensure manual regions are applied
    if not app.auto_detect_crossing:
        if app.lane_region and app.detector:
            app.detector.lane_region = app.lane_region
        if app.crossing_region and app.detector:
            app.detector.set_manual_crossing(app.crossing_region)
    
    # Main detection
    detections, annotated = app.detector.detect(frame)
    
    # Get lane vehicles and pedestrians from main camera
    lane_vehicles = [d for d in detections if d.type in [DetectionType.VEHICLE, DetectionType.MOTORCYCLE] and d.in_lane]
    pedestrians_main = [d for d in detections if d.type == DetectionType.PEDESTRIAN and d.at_crossing]
    
    # Update pedestrian tracker if enabled
    pedestrian_stats = {}
    if app.pedestrian_tracker_enabled and app.detector.pedestrian_tracker:
        tracked_peds, pedestrian_stats = app.detector.pedestrian_tracker.update(detections)
        if pedestrian_stats.get('waiting_count', 0) > 0:
            app.controller.pedestrians_waiting_count = pedestrian_stats['waiting_count']
    
    # Update controller
    phase = app.controller.update_detections(lane_vehicles, pedestrians_main)
    
    # Update stats history
    stats = app.detector.get_stats()
    app.stats_history.append({
        'timestamp': time.time(),
        'blocking': len(lane_vehicles),
        'pedestrians': pedestrian_stats.get('waiting_count', len(pedestrians_main)),
        'fps': stats['fps']
    })
    if len(app.stats_history) > 100:
        app.stats_history.pop(0)
    
    app.pedestrian_stats = pedestrian_stats
    
    # ========== FIXED: Conditional column layout ==========
    # Check if we have pedestrian camera feed to display
    has_ped_feed = ped_frame is not None and app.pedestrian_tracker_enabled and app.detector.pedestrian_tracker
    
    if has_ped_feed:
        # Dual camera mode: show both feeds side by side
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(annotated, channels="BGR", use_container_width=True, caption="Main Camera Feed")
        
        with col2:
            ped_annotated = app.detector.pedestrian_tracker.draw(ped_frame)
            st.image(ped_annotated, channels="BGR", use_container_width=True, caption="Pedestrian Camera Feed")
    else:
        # Single camera mode: use full width
        st.image(annotated, channels="BGR", use_container_width=True, caption="Live Detection Feed")
    
    # Detection info panels (always show)
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("### 🚗 Free-Left Lane")
        if lane_vehicles:
            st.warning(f"⚠️ {len(lane_vehicles)} vehicle(s) in lane")
            for v in lane_vehicles[:3]:
                icon = "🏍️" if v.type == DetectionType.MOTORCYCLE else "🚗"
                blocking = "🔴 BLOCKING" if v.is_stationary and v.stationary_time > 3 else "🟡 In lane"
                st.write(f"{icon} {v.class_name} - {blocking} ({v.stationary_time:.1f}s)")
        else:
            st.success("✅ Lane clear")
    
    with info_col2:
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
    
    with info_col3:
        st.markdown("### 📊 Detection Stats")
        st.metric("FPS", f"{stats['fps']:.1f}")
        st.metric("Vehicles", stats['tracked_vehicles'])
        st.metric("Motorcycles", stats.get('tracked_motorcycles', 0))
        st.metric("Pedestrians", stats['tracked_pedestrians'])
        
        if app.crossing_region:
            st.success("✓ Manual crossing")
        elif stats.get('crossing_detected', False):
            st.success(f"✓ Auto crossing")
        else:
            st.warning("⚠ No crossing region")
    
    time.sleep(0.03)
    st.rerun()


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
            st.subheader("⚙️ Detection Settings")
            
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
            st.subheader("📹 Camera Configuration")
            
            camera_layout = st.radio(
                "Camera Setup",
                ["Single Camera", "Dual Camera (Main + Pedestrian)"],
                help="Choose camera setup for monitoring"
            )
            st.session_state.app_state.dual_camera_mode = (camera_layout == "Dual Camera (Main + Pedestrian)")
            
            # Main Camera Source
            source = render_video_source()
            
            # Pedestrian Camera Source (if dual camera mode)
            ped_source = None
            if st.session_state.app_state.dual_camera_mode:
                st.divider()
                ped_source = render_pedestrian_camera()
                
                # Pedestrian region configuration
                st.divider()
                waiting_region, crossing_region = render_pedestrian_region_config()
            
            st.divider()
            
            # Manual Region Configuration (only when auto-detect is OFF)
            if not auto_detect:
                st.subheader("📍 Manual Region Setup")
                st.warning("Auto-detection is OFF - Define regions manually below")
                
                width = st.session_state.app_state.frame_width
                height = st.session_state.app_state.frame_height
                
                # Lane Region
                st.markdown("### 🟡 Free-Left Lane")
                lane_x1 = st.number_input("Lane Left X", 0, width, width // 10, key="sidebar_lane_x1")
                lane_y1 = st.number_input("Lane Top Y", 0, height, height - 150, key="sidebar_lane_y1")
                lane_x2 = st.number_input("Lane Right X", 0, width, width // 3, key="sidebar_lane_x2")
                lane_y2 = st.number_input("Lane Bottom Y", 0, height, height - 120, key="sidebar_lane_y2")
                lane_region = [(lane_x1, lane_y1), (lane_x2, lane_y1), (lane_x2, lane_y2), (lane_x1, lane_y2)]
                
                # Crossing Region
                st.markdown("### 🟢 Pedestrian Crossing")
                cross_x1 = st.number_input("Crossing Left X", 0, width, width // 3, key="sidebar_cross_x1")
                cross_y1 = st.number_input("Crossing Top Y", 0, height, height // 2 - 50, key="sidebar_cross_y1")
                cross_x2 = st.number_input("Crossing Right X", 0, width, 2 * width // 3, key="sidebar_cross_x2")
                cross_y2 = st.number_input("Crossing Bottom Y", 0, height, height // 2 + 50, key="sidebar_cross_y2")
                crossing_region = [(cross_x1, cross_y1), (cross_x2, cross_y1), (cross_x2, cross_y2), (cross_x1, cross_y2)]
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Apply Lane", key="sidebar_apply_lane", use_container_width=True):
                        st.session_state.app_state.lane_region = lane_region
                        st.session_state.lane_region_set = True
                        if st.session_state.app_state.detector:
                            st.session_state.app_state.detector.lane_region = lane_region
                        st.success("Lane region applied")
                
                with col2:
                    if st.button("✅ Apply Crossing", key="sidebar_apply_crossing", use_container_width=True):
                        st.session_state.app_state.crossing_region = crossing_region
                        st.session_state.crossing_region_set = True
                        if st.session_state.app_state.detector:
                            st.session_state.app_state.detector.set_manual_crossing(crossing_region)
                        st.success("Crossing region applied")
                
                # Mini preview
                if lane_region and crossing_region:
                    preview = np.zeros((200, 300, 3), dtype=np.uint8)
                    scale_x = 300 / width
                    scale_y = 200 / height
                    
                    scaled_lane = [(int(x * scale_x), int(y * scale_y)) for (x, y) in lane_region]
                    scaled_cross = [(int(x * scale_x), int(y * scale_y)) for (x, y) in crossing_region]
                    
                    cv2.polylines(preview, [np.array(scaled_lane, dtype=np.int32)], True, (0, 255, 255), 2)
                    cv2.polylines(preview, [np.array(scaled_cross, dtype=np.int32)], True, (0, 255, 0), 2)
                    st.image(preview, caption="Region Preview", use_container_width=True)
            
            st.divider()
            
            # Start/Stop buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ START", type="primary", use_container_width=True):
                    if source:
                        app = st.session_state.app_state
                        
                        # Create main detector
                        app.detector = UnifiedDetector(
                            auto_detect_crossing=auto_detect,
                            conf_threshold=confidence,
                            enhance_visibility=enhance_vis,
                            debug_mode=debug_mode
                        )
                        
                        # Apply manual regions if auto-detect is off
                        if not auto_detect:
                            if app.lane_region:
                                app.detector.lane_region = app.lane_region
                            if app.crossing_region:
                                app.detector.set_manual_crossing(app.crossing_region)
                        
                        # Enable pedestrian tracking if configured
                        if st.session_state.app_state.dual_camera_mode and st.session_state.app_state.pedestrian_region:
                            ped_region = st.session_state.app_state.pedestrian_region
                            app.detector.enable_pedestrian_tracking(
                                crossing_region=ped_region['crossing'],
                                waiting_region=ped_region.get('waiting')
                            )
                            app.pedestrian_tracker_enabled = True
                        
                        # Create main video source
                        if source['type'] == 'camera':
                            app.video_source = cv2.VideoCapture(source['source'])
                        else:
                            app.video_source = cv2.VideoCapture(source['source'])
                        
                        # Create pedestrian video source if in dual camera mode
                        if app.dual_camera_mode and ped_source and ped_source['type'] != 'same':
                            if ped_source['type'] == 'camera':
                                app.pedestrian_camera = cv2.VideoCapture(ped_source['source'])
                            elif ped_source['type'] == 'video':
                                app.pedestrian_camera = cv2.VideoCapture(ped_source['source'])
                        
                        if app.video_source.isOpened():
                            app.is_running = True
                            mode_text = "Manual Regions" if not auto_detect else "Auto Detection"
                            cam_text = "Dual Camera" if app.dual_camera_mode else "Single Camera"
                            st.success(f"✅ Started! Mode: {mode_text} | {cam_text}")
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
                    app.detector = None
                    app.video_source = None
                    app.pedestrian_camera = None
                    st.info("Stopped")
                    st.rerun()
            
            st.divider()
            
            # Manual override
            st.subheader("🔧 Manual Override")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔒 Protected Left"):
                    st.session_state.app_state.controller.manual_protect()
                    st.success("Protected left activated")
            with col2:
                if st.button("🚶 Pedestrian Mode"):
                    st.session_state.app_state.controller.manual_pedestrian()
                    st.success("Pedestrian mode activated")
            
            if st.button("🔄 Reset to Auto"):
                st.session_state.app_state.controller.manual_reset()
                st.info("Auto mode restored")
        
        else:
            # Dataset analysis mode
            st.subheader("📄 Upload Dataset")
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
        
        return mode


def render_analytics():
    """Render analytics dashboard"""
    app = st.session_state.app_state
    
    st.subheader("📊 System Analytics")
    
    if app.stats_history:
        df = pd.DataFrame(app.stats_history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['blocking'], name='Blocking Vehicles', mode='lines', line=dict(color='red')))
        fig.add_trace(go.Scatter(y=df['pedestrians'], name='Pedestrians', mode='lines', line=dict(color='blue')))
        fig.update_layout(title='Real-time Metrics', xaxis_title='Time (frames)', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    
    # Pedestrian stats if available
    if app.pedestrian_stats:
        st.markdown("### 🚶 Pedestrian Analytics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Crossed Today", app.pedestrian_stats.get('total_crossed', 0))
        with col2:
            st.metric("Max Waiting", app.pedestrian_stats.get('max_waiting', 0))
        with col3:
            st.metric("Peak Hour Peds", app.pedestrian_stats.get('peak_hour_pedestrians', 0))
        with col4:
            st.metric("Avg Crossing Time", f"{app.pedestrian_stats.get('avg_crossing_time', 0):.1f}s")
    
    # Events log
    events = app.controller.get_events(100)
    if events:
        with st.expander("📋 Event Log"):
            for event in reversed(events[-20:]):
                ts = event['timestamp'][:19]
                if event['event'] == 'PROTECTED_TRIGGERED':
                    st.warning(f"**{ts}** - 🟢 PROTECTED LEFT | {event['data'].get('violations', 0)} vehicles | {event['data'].get('reason', '')}")
                elif event['event'] == 'PEDESTRIAN_MODE':
                    st.info(f"**{ts}** - 🚶 PEDESTRIAN MODE | {event['data'].get('count', 0)} waiting, {event['data'].get('waiting_time', 0):.1f}s")
                elif event['event'] == 'EMERGENCY_STOP':
                    st.error(f"**{ts}** - 🔴 EMERGENCY STOP | {event['data'].get('conflicts', 0)} conflicts")
                else:
                    st.write(f"**{ts}** - {event['event']}")


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
    
    recs = results.get('recommendations', [])
    if recs:
        st.markdown("### 💡 Recommendations")
        for rec in recs[:5]:
            st.info(rec)


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
    
    st.divider()
    st.caption(f"🕒 System Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Status: {'🟢 Active' if st.session_state.app_state.is_running else '⚪ Idle'} | Dual Camera: {'✅' if st.session_state.app_state.dual_camera_mode else '❌'}")


if __name__ == "__main__":
    main()