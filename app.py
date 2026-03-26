#!/usr/bin/env python3
"""
Intelligent Free Left Turn Management System
NO BLINKING VERSION - Uses Streamlit's native video display with periodic updates
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
import threading
import queue
from typing import Optional, Dict, List, Tuple

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
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .signal-card {
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .signal-free { background: linear-gradient(135deg, #ffa500, #ff8c00); color: white; }
    .signal-protected { background: linear-gradient(135deg, #00cc66, #00994d); color: white; }
    .signal-pedestrian { background: linear-gradient(135deg, #ff4b4b, #cc0000); color: white; }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #1e3c72; }
    .metric-label { font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }
    .footer { text-align: center; padding: 1rem; margin-top: 2rem; border-top: 1px solid #e9ecef; color: #6c757d; font-size: 0.75rem; }
    .stImage { margin-bottom: 0 !important; }
    .video-container {
        position: relative;
        width: 100%;
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORTS
# ============================================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import UnifiedDetector, DetectionType
from core.signal_controller import EnhancedSignalController, SignalPhase
from core.pdf_parser import PDFParser
from core.data_analyzer import DataAnalyzer


# ============================================================================
# APP STATE
# ============================================================================
class AppState:
    """Application state management"""
    def __init__(self):
        self.detector = None
        self.controller = EnhancedSignalController()
        self.video_source = None
        self.pedestrian_source = None
        self.is_running = False
        self.analysis_results = None
        self.intersection_name = "Banashankari Junction"
        self.lane_region = None
        self.crossing_region = None
        self.auto_detect_crossing = True
        self.enhance_visibility = True
        self.stats_history = []
        self.frame_width = 1280
        self.frame_height = 720
        self.dual_camera_mode = False
        self.current_frame = None
        self.current_ped_frame = None
        self.current_stats = {}
        self.last_update_time = 0
        self.update_interval = 0.05  # 20 FPS


def init_session_state():
    """Initialize session state"""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'frame_counter' not in st.session_state:
        st.session_state.frame_counter = 0


# ============================================================================
# PROCESSING FUNCTION
# ============================================================================
def process_single_frame(app, frame):
    """Process a single frame - returns annotated frame and stats"""
    if frame is None or app.detector is None:
        return None, {}
    
    try:
        # Apply manual regions
        if not app.auto_detect_crossing:
            if app.lane_region and app.detector:
                app.detector.lane_region = app.lane_region
            if app.crossing_region and app.detector:
                app.detector.set_manual_crossing(app.crossing_region)
        
        # Detect
        detections, annotated = app.detector.detect(frame)
        
        # Get lane vehicles and pedestrians
        lane_vehicles = []
        pedestrians = []
        
        for d in detections:
            if d.type in [DetectionType.VEHICLE, DetectionType.MOTORCYCLE] and d.in_lane:
                lane_vehicles.append(d)
            elif d.type == DetectionType.PEDESTRIAN and d.at_crossing:
                pedestrians.append(d)
        
        # Update controller
        phase = app.controller.update_detections(lane_vehicles, pedestrians)
        
        # Get stats
        stats = app.detector.get_stats()
        stats['blocking_count'] = len(lane_vehicles)
        stats['pedestrian_count'] = len(pedestrians)
        
        return annotated, stats
        
    except Exception as e:
        print(f"Processing error: {e}")
        return frame, {}


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🚦 Intelligent Free Left Turn Management System</h1>
        <p>AI-Powered Traffic Management | Real-time Detection | Pedestrian Safety</p>
    </div>
    """, unsafe_allow_html=True)


def render_signal_status():
    """Render current signal status"""
    app = st.session_state.app_state
    state = app.controller.get_state()
    
    st.markdown("### 🚥 Current Signal Status")
    
    phase = state['phase']
    
    if "FREE" in phase:
        st.markdown(f"""
        <div class="signal-card signal-free">
            <h2 style="margin: 0;">{phase}</h2>
            <p style="margin: 0.5rem 0 0 0;">Yield to oncoming traffic</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PROTECTED" in phase:
        st.markdown(f"""
        <div class="signal-card signal-protected">
            <h2 style="margin: 0;">{phase}</h2>
            <p style="margin: 0.5rem 0 0 0;">Green arrow - Safe to turn</p>
        </div>
        """, unsafe_allow_html=True)
    elif "PEDESTRIAN" in phase:
        st.markdown(f"""
        <div class="signal-card signal-pedestrian">
            <h2 style="margin: 0;">{phase}</h2>
            <p style="margin: 0.5rem 0 0 0;">🚶 Pedestrians crossing - STOP</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{state['blocking_vehicles']}</div>
            <div class="metric-label">🚗 Blocking</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{state.get('pedestrians_waiting', 0)}</div>
            <div class="metric-label">🚶 Pedestrians</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{state['phase_duration']:.0f}s</div>
            <div class="metric-label">⏱️ Duration</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{state['protected_triggers']}</div>
            <div class="metric-label">🔄 Interventions</div>
        </div>
        """, unsafe_allow_html=True)
    
    if state.get('pedestrian_mode_active', False):
        st.info(f"🚶 Pedestrian Mode Active - {state.get('pedestrians_waiting', 0)} pedestrians waiting")
    
    if state.get('cooldown_remaining', 0) > 0:
        st.progress(min(state['cooldown_remaining'] / 30, 1.0))
        st.caption(f"Cooldown: {state['cooldown_remaining']:.0f}s remaining")


def render_analytics():
    """Render analytics dashboard"""
    app = st.session_state.app_state
    
    st.markdown("### 📊 System Analytics")
    
    if app.stats_history:
        df = pd.DataFrame(app.stats_history[-100:])
        if not df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=df['blocking'], name='Blocking', mode='lines', line=dict(color='#ff4b4b', width=2)))
            fig.add_trace(go.Scatter(y=df['pedestrians'], name='Pedestrians', mode='lines', line=dict(color='#17a2b8', width=2)))
            fig.update_layout(title='Real-time Metrics', xaxis_title='Time', yaxis_title='Count', height=300)
            st.plotly_chart(fig, use_container_width=True, key="analytics_chart")
    
    events = app.controller.get_events(50)
    if events:
        with st.expander("📋 Event Log", expanded=False):
            for event in reversed(events[-10:]):
                ts = time.strftime('%H:%M:%S', time.localtime(event['timestamp'])) if isinstance(event['timestamp'], (int, float)) else event['timestamp'][:19]
                if event['event'] == 'PROTECTED_TRIGGERED':
                    st.warning(f"**{ts}** - 🟢 PROTECTED LEFT")
                elif event['event'] == 'PEDESTRIAN_MODE':
                    st.info(f"**{ts}** - 🚶 PEDESTRIAN MODE")
                else:
                    st.write(f"**{ts}** - {event['event']}")


def render_manual_region_config():
    """Render manual region configuration"""
    st.subheader("🎯 Manual Region Configuration")
    
    app = st.session_state.app_state
    width, height = app.frame_width, app.frame_height
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟡 FREE LEFT LANE")
        lane_x1 = st.number_input("Left X", 0, width, width//10, key="lane_x1")
        lane_y1 = st.number_input("Top Y", 0, height, height-150, key="lane_y1")
        lane_x2 = st.number_input("Right X", 0, width, width//3, key="lane_x2")
        lane_y2 = st.number_input("Bottom Y", 0, height, height-120, key="lane_y2")
        lane_region = [(lane_x1, lane_y1), (lane_x2, lane_y1), (lane_x2, lane_y2), (lane_x1, lane_y2)]
        
        if st.button("✅ Apply Lane", key="apply_lane"):
            app.lane_region = lane_region
            if app.detector:
                app.detector.lane_region = lane_region
            st.success("Lane applied")
    
    with col2:
        st.markdown("### 🟢 PEDESTRIAN CROSSING")
        cross_x1 = st.number_input("Left X", 0, width, width//3, key="cross_x1")
        cross_y1 = st.number_input("Top Y", 0, height, height//2-50, key="cross_y1")
        cross_x2 = st.number_input("Right X", 0, width, 2*width//3, key="cross_x2")
        cross_y2 = st.number_input("Bottom Y", 0, height, height//2+50, key="cross_y2")
        crossing_region = [(cross_x1, cross_y1), (cross_x2, cross_y1), (cross_x2, cross_y2), (cross_x1, cross_y2)]
        
        if st.button("✅ Apply Crossing", key="apply_crossing"):
            app.crossing_region = crossing_region
            if app.detector:
                app.detector.set_manual_crossing(crossing_region)
            st.success("Crossing applied")
    
    return lane_region, crossing_region


def render_video_source(key="main"):
    """Render video source selection"""
    st.subheader(f"📹 {key.capitalize()} Source")
    
    source_type = st.radio(
        "Source Type",
        ["📷 Live Camera", "🎬 Video File"],
        horizontal=True,
        key=f"source_type_{key}"
    )
    
    if source_type == "📷 Live Camera":
        camera_id = st.selectbox("Camera", ["Webcam (0)", "External (1)"], index=0, key=f"camera_{key}")
        camera_index = int(camera_id.split("(")[1].split(")")[0])
        
        test_cap = cv2.VideoCapture(camera_index)
        if test_cap.isOpened():
            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if key == "main":
                st.session_state.app_state.frame_width = width
                st.session_state.app_state.frame_height = height
            test_cap.release()
        
        return {'type': 'camera', 'source': camera_index, 'name': f'Camera {camera_index}'}
    else:
        uploaded = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], key=f"video_{key}")
        if uploaded:
            temp_path = os.path.join(tempfile.gettempdir(), f"{key}_video_{int(time.time())}.mp4")
            with open(temp_path, 'wb') as f:
                f.write(uploaded.getbuffer())
            
            cap = cv2.VideoCapture(temp_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if key == "main":
                st.session_state.app_state.frame_width = width
                st.session_state.app_state.frame_height = height
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Resolution", f"{width}x{height}")
            with col2:
                st.metric("Duration", f"{frame_count/fps:.1f}s")
            
            return {'type': 'video', 'source': temp_path, 'name': uploaded.name}
        return None


def render_realtime_feed():
    """Render real-time detection feed - NO BLINKING"""
    app = st.session_state.app_state
    
    if not app.is_running or not app.video_source:
        st.info("📹 No active source. Configure settings and click START.")
        return
    
    # Read frames
    ret, frame = app.video_source.read()
    if not ret:
        if hasattr(app.video_source, 'set'):
            app.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = app.video_source.read()
        if not ret:
            st.warning("End of video")
            return
    
    # Read pedestrian frame if available
    ped_frame = None
    if app.dual_camera_mode and app.pedestrian_source:
        ret_ped, ped_frame = app.pedestrian_source.read()
        if not ret_ped and hasattr(app.pedestrian_source, 'set'):
            app.pedestrian_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret_ped, ped_frame = app.pedestrian_source.read()
    
    # Process frame
    annotated, stats = process_single_frame(app, frame)
    
    # Process pedestrian frame if available
    ped_annotated = None
    if ped_frame is not None and app.detector:
        ped_annotated, _ = process_single_frame(app, ped_frame)
    
    # Update stats history
    app.stats_history.append({
        'timestamp': time.time(),
        'blocking': stats.get('blocking_count', 0),
        'pedestrians': stats.get('pedestrian_count', 0),
        'fps': stats.get('fps', 0)
    })
    if len(app.stats_history) > 100:
        app.stats_history.pop(0)
    
    # Create layout
    if app.dual_camera_mode:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if annotated is not None:
                st.image(annotated, channels="BGR", use_container_width=True)
        
        with col2:
            if ped_annotated is not None:
                st.image(ped_annotated, channels="BGR", use_container_width=True, caption="Pedestrian Camera")
            else:
                st.info("📹 Pedestrian camera not available")
    else:
        if annotated is not None:
            st.image(annotated, channels="BGR", use_container_width=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🚗 Free-Left Lane")
        blocking_count = stats.get('blocking_count', 0)
        if blocking_count > 0:
            st.warning(f"⚠️ {blocking_count} vehicle(s) in lane")
        else:
            st.success("✅ Lane clear")
    
    with col2:
        st.markdown("### 🚶 Pedestrian Status")
        ped_count = stats.get('pedestrian_count', 0)
        if ped_count > 0:
            st.info(f"🚶 {ped_count} pedestrian(s) at crossing")
        else:
            st.success("✅ No pedestrians")
    
    with col3:
        st.markdown("### 📊 Detection Stats")
        st.metric("FPS", f"{stats.get('fps', 0):.1f}")
        st.metric("Vehicles", stats.get('tracked_vehicles', 0))
        st.metric("Pedestrians", stats.get('tracked_pedestrians', 0))
    
    # Small delay - DO NOT USE st.rerun()
    time.sleep(0.03)


def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.markdown("## 🎮 System Controls")
        
        mode = st.radio("Select Mode", ["📊 Dataset Analysis", "📹 Real-time Monitoring"])
        st.divider()
        
        if mode == "📹 Real-time Monitoring":
            st.markdown("### ⚙️ Settings")
            
            auto_detect = st.checkbox("🤖 Auto-Detect Zebra Crossing", value=st.session_state.app_state.auto_detect_crossing)
            st.session_state.app_state.auto_detect_crossing = auto_detect
            
            enhance_vis = st.checkbox("✨ Enhance Visibility", value=st.session_state.app_state.enhance_visibility)
            st.session_state.app_state.enhance_visibility = enhance_vis
            
            confidence = st.slider("Detection Confidence", 0.3, 0.8, 0.4, 0.05)
            
            st.divider()
            
            # Camera Setup
            st.markdown("### 📹 Camera Setup")
            
            camera_mode = st.radio("Camera Mode", ["Single Camera", "Dual Camera"], horizontal=True)
            st.session_state.app_state.dual_camera_mode = (camera_mode == "Dual Camera")
            
            # Main Camera
            main_source = render_video_source("main")
            
            # Pedestrian Camera (if dual mode)
            ped_source = None
            if st.session_state.app_state.dual_camera_mode:
                st.divider()
                ped_source = render_video_source("pedestrian")
            
            st.divider()
            
            # Manual Regions (if auto-detect off)
            if not auto_detect:
                lane_region, crossing_region = render_manual_region_config()
                st.session_state.app_state.lane_region = lane_region
                st.session_state.app_state.crossing_region = crossing_region
            
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
                            debug_mode=False
                        )
                        
                        if not auto_detect:
                            if app.lane_region:
                                app.detector.lane_region = app.lane_region
                            if app.crossing_region:
                                app.detector.set_manual_crossing(app.crossing_region)
                        
                        if main_source['type'] == 'camera':
                            app.video_source = cv2.VideoCapture(main_source['source'])
                        else:
                            app.video_source = cv2.VideoCapture(main_source['source'])
                        
                        if ped_source:
                            if ped_source['type'] == 'camera':
                                app.pedestrian_source = cv2.VideoCapture(ped_source['source'])
                            else:
                                app.pedestrian_source = cv2.VideoCapture(ped_source['source'])
                        
                        if app.video_source.isOpened():
                            app.is_running = True
                            st.success("✅ Started!")
                            # No rerun - just start processing
                        else:
                            st.error("Cannot open source")
                    else:
                        st.warning("Select a source first")
            
            with col2:
                if st.button("⏹️ STOP", use_container_width=True):
                    app = st.session_state.app_state
                    app.is_running = False
                    if app.video_source:
                        app.video_source.release()
                    if app.pedestrian_source:
                        app.pedestrian_source.release()
                    app.detector = None
                    app.video_source = None
                    app.pedestrian_source = None
                    st.info("Stopped")
            
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
                st.info("Auto mode restored")
        
        else:
            # Dataset Analysis
            st.markdown("### 📄 Upload Dataset")
            uploaded = st.file_uploader("Upload Traffic Data PDF", type=['pdf'])
            
            if uploaded:
                if st.button("🔍 Analyze Dataset", type="primary", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        try:
                            temp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
                            with open(temp_path, 'wb') as f:
                                f.write(uploaded.getbuffer())
                            
                            parser = PDFParser()
                            data = parser.parse(temp_path)
                            analyzer = DataAnalyzer(data)
                            analysis = analyzer.analyze()
                            
                            st.session_state.app_state.analysis_results = analysis
                            st.session_state.analysis_complete = True
                            st.session_state.app_state.controller.integrate_dataset(analysis)
                            st.success("✅ Analysis complete!")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            
            if st.session_state.analysis_complete and st.session_state.app_state.analysis_results:
                results = st.session_state.app_state.analysis_results
                risk = results.get('risk_assessment', {})
                st.markdown(f"""
                <div style="background: {'#ff4b4b' if risk.get('level') == 'HIGH' else '#ffa500' if risk.get('level') == 'MEDIUM' else '#00cc66'}; color: white; padding: 0.5rem; border-radius: 8px; text-align: center;">
                    <h4>Risk: {risk.get('level', 'UNKNOWN')} ({risk.get('score', 0)}/100)</h4>
                </div>
                """, unsafe_allow_html=True)
        
        return mode


def render_dataset_results():
    """Render dataset analysis results"""
    results = st.session_state.app_state.analysis_results
    if not results:
        return
    
    st.markdown("### 📊 Dataset Analysis Results")
    
    risk = results.get('risk_assessment', {})
    st.markdown(f"**Risk Level:** {risk.get('level', 'UNKNOWN')} (Score: {risk.get('score', 0)}/100)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Violations", results.get('violation_analysis', {}).get('total_violations', 0))
    with col2:
        peak = results.get('violation_analysis', {}).get('peak_hours', {})
        if peak:
            st.metric("Peak Hours", ', '.join([f"{h}:00" for h in peak.keys()]))
    
    recs = results.get('recommendations', [])
    if recs:
        with st.expander("💡 Recommendations"):
            for rec in recs[:5]:
                st.info(rec)


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main application entry point"""
    init_session_state()
    render_header()
    
    mode = render_sidebar()
    
    if mode == "📹 Real-time Monitoring":
        # Use auto-refresh pattern without st.rerun()
        with st.container():
            render_realtime_feed()
        
        with st.container():
            col1, col2 = st.columns([2, 1])
            with col1:
                render_signal_status()
            with col2:
                render_analytics()
        
        # Auto-refresh using time.sleep in a loop (handled by render_realtime_feed)
        if st.session_state.app_state.is_running:
            time.sleep(0.05)
            st.rerun()
    
    else:
        if st.session_state.analysis_complete and st.session_state.app_state.analysis_results:
            render_dataset_results()
        else:
            st.info("📄 Upload a PDF dataset from the sidebar to begin analysis")
        render_analytics()
    
    st.markdown("""
    <div class="footer">
        <p>🚦 Intelligent Free Left Turn Management System | AI-Powered Traffic Management</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()