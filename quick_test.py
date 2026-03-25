#!/usr/bin/env python3
"""
Quick test script - Fast verification with single video
"""

import sys
import os
import cv2
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import UnifiedDetector
from core.signal_controller import EnhancedSignalController


def quick_test(video_path, max_frames=300):
    """
    Quick test with limited frames for fast verification
    """
    print("\n" + "="*60)
    print(f"🚦 QUICK TEST: {video_path}")
    print("="*60)
    
    # Initialize
    detector = UnifiedDetector(auto_detect_crossing=True, conf_threshold=0.4)
    controller = EnhancedSignalController()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"\n📹 Processing first {max_frames} frames at {fps:.1f} fps...\n")
    
    frame_count = 0
    start_time = time.time()
    
    # Statistics
    vehicle_count = 0
    pedestrian_count = 0
    blocking_count = 0
    protected_triggers = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections, annotated = detector.detect(frame)
        
        # Count
        vehicles = [d for d in detections if hasattr(d, 'type') and d.type.value != 'pedestrian']
        pedestrians = [d for d in detections if hasattr(d, 'type') and d.type.value == 'pedestrian']
        lane_vehicles = [d for d in detections if hasattr(d, 'in_lane') and d.in_lane]
        blocking = [d for d in lane_vehicles if hasattr(d, 'is_stationary') and d.is_stationary]
        
        vehicle_count += len(vehicles)
        pedestrian_count += len(pedestrians)
        blocking_count += len(blocking)
        
        # Update controller
        old_phase = controller.current_phase.value
        new_phase = controller.update_detections(lane_vehicles, pedestrians)
        
        if old_phase != new_phase.value and "PROTECTED" in new_phase.value:
            protected_triggers += 1
            print(f"  Frame {frame_count}: {old_phase} → {new_phase.value}")
        
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            print(f"  Processed {frame_count} frames...")
    
    cap.release()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print("📊 QUICK TEST RESULTS")
    print("="*60)
    print(f"\n📹 Processed: {frame_count} frames in {elapsed:.2f}s")
    print(f"   Speed: {frame_count/elapsed:.1f} fps")
    print(f"\n🚗 Detections:")
    print(f"   Total Vehicles: {vehicle_count}")
    print(f"   Total Pedestrians: {pedestrian_count}")
    print(f"   Blocking Events: {blocking_count}")
    print(f"\n🚦 Signal Changes:")
    print(f"   Protected Left Triggers: {protected_triggers}")
    
    # Display final state
    state = controller.get_state()
    print(f"\n📊 Final State:")
    print(f"   Phase: {state['phase']}")
    print(f"   Risk Level: {state['risk_level']}")
    print(f"   Blocking Vehicles: {state['blocking_vehicles']}")
    
    print("\n✅ Quick test completed!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        quick_test(sys.argv[1])
    else:
        # Check for demo video
        if os.path.exists("demo_traffic.mp4"):
            quick_test("demo_traffic.mp4")
        else:
            print("Please provide a video file or run create_test_videos.py first")
            print("\nUsage: python quick_test.py path/to/video.mp4")