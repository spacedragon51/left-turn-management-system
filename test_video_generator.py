#!/usr/bin/env python3
"""
Create comprehensive test videos for dry run
Generates videos with vehicles, pedestrians, and zebra crossings
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

def create_test_video_with_traffic(output_file="test_traffic.mp4", duration=30, fps=30):
    """
    Create test video with moving vehicles and pedestrian crossing
    """
    print(f"🎬 Creating test video: {output_file}")
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    # Define regions
    lane_region = [(width//10, height-150), (width//3, height-150), 
                   (width//3+30, height-120), (width//10, height-120)]
    crossing_region = [(width//3, height//2-50), (2*width//3, height//2-50),
                       (2*width//3, height//2+50), (width//3, height//2+50)]
    
    for frame_num in range(total_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Sky
        cv2.rectangle(frame, (0, 0), (width, height-250), (135, 206, 235), -1)
        
        # Road
        cv2.rectangle(frame, (0, height-250), (width, height), (50, 50, 50), -1)
        
        # Lane markings
        for i in range(0, width, 50):
            cv2.rectangle(frame, (i, height-125), (i+30, height-123), (255, 255, 255), -1)
        
        # Free-left lane boundary
        pts = np.array(lane_region, np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        cv2.putText(frame, "FREE LEFT LANE", (lane_region[0][0], lane_region[0][1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Zebra crossing
        x1, y1 = crossing_region[0]
        x2, y2 = crossing_region[2]
        stripe_width = (x2 - x1) // 8
        for i in range(8):
            x = x1 + i * stripe_width
            if i % 2 == 0:
                cv2.rectangle(frame, (x, y1), (x + stripe_width, y2), (255, 255, 255), -1)
        cv2.polylines(frame, [np.array(crossing_region, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, "PEDESTRIAN CROSSING", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Moving vehicles (oncoming traffic)
        t = frame_num / fps
        
        # Vehicle 1: moving left to right
        x1_pos = (frame_num * 5) % (width + 200) - 200
        if 0 <= x1_pos <= width - 80:
            cv2.rectangle(frame, (int(x1_pos), height-190), (int(x1_pos)+70, height-145), (0, 255, 0), -1)
            cv2.putText(frame, "Car", (int(x1_pos)+25, height-165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Vehicle 2: moving right to left
        x2_pos = width - (frame_num * 4) % (width + 200)
        if 0 <= x2_pos <= width - 80:
            cv2.rectangle(frame, (int(x2_pos), height-190), (int(x2_pos)+70, height-145), (255, 0, 0), -1)
            cv2.putText(frame, "Oncoming", (int(x2_pos)+15, height-165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Blocking vehicles in free-left lane
        # Scenario 1: First blocking event (5-10 seconds)
        if 5 < t < 10:
            cv2.rectangle(frame, (200, height-145), (270, height-105), (0, 0, 255), -1)
            cv2.putText(frame, "BLOCKING!", (210, height-120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "⚠️ VEHICLE BLOCKING FREE LEFT LANE", (width//2-200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Scenario 2: Multiple blocking vehicles (15-20 seconds)
        if 15 < t < 20:
            cv2.rectangle(frame, (180, height-145), (250, height-105), (0, 0, 255), -1)
            cv2.putText(frame, "BLOCKING!", (190, height-120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.rectangle(frame, (270, height-145), (340, height-105), (0, 0, 255), -1)
            cv2.putText(frame, "BLOCKING!", (280, height-120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "🚨 MULTIPLE VEHICLES BLOCKING - PROTECTED LEFT TRIGGERED", (width//2-250, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Scenario 3: Pedestrians waiting (22-28 seconds)
        if 22 < t < 28:
            for i in range(3):
                ped_x = width//2 + (i-1)*40
                ped_y = height//2 - 30
                cv2.circle(frame, (ped_x, ped_y), 12, (0, 0, 255), -1)
                cv2.rectangle(frame, (ped_x-5, ped_y+5), (ped_x+5, ped_y+20), (0, 0, 255), -1)
                cv2.putText(frame, "WAITING", (ped_x-20, ped_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(frame, "🚶 PEDESTRIANS WAITING - ACTIVATING PEDESTRIAN MODE", (width//2-300, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add timestamp
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        
        if frame_num % (fps * 5) == 0:
            print(f"  Progress: {frame_num/fps:.0f}/{duration}s")
    
    out.release()
    print(f"✅ Test video created: {output_file}")
    return output_file


def create_pedestrian_test_video(output_file="test_pedestrians.mp4", duration=20, fps=30):
    """
    Create test video focused on pedestrian crossing
    """
    print(f"🎬 Creating pedestrian test video: {output_file}")
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    crossing_region = [(width//3, height//2-50), (2*width//3, height//2-50),
                       (2*width//3, height//2+50), (width//3, height//2+50)]
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(frame, (0, 0), (width, height-200), (135, 206, 235), -1)
        cv2.rectangle(frame, (0, height-200), (width, height), (50, 50, 50), -1)
        
        # Zebra crossing
        x1, y1 = crossing_region[0]
        x2, y2 = crossing_region[2]
        stripe_width = (x2 - x1) // 8
        for i in range(8):
            x = x1 + i * stripe_width
            if i % 2 == 0:
                cv2.rectangle(frame, (x, y1), (x + stripe_width, y2), (255, 255, 255), -1)
        cv2.polylines(frame, [np.array(crossing_region, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, "PEDESTRIAN CROSSING", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        t = frame_num / fps
        
        # Pedestrians waiting
        if 3 < t < 8:
            for i in range(3):
                ped_x = width//2 + (i-1)*50
                ped_y = height//2 - 30
                cv2.circle(frame, (ped_x, ped_y), 12, (0, 0, 255), -1)
                cv2.rectangle(frame, (ped_x-5, ped_y+5), (ped_x+5, ped_y+20), (0, 0, 255), -1)
                cv2.putText(frame, f"P{i+1}", (ped_x-5, ped_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"🚶 {3} PEDESTRIANS WAITING - {t-3:.0f}s", (width//2-150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Pedestrians crossing
        if 10 < t < 15:
            for i in range(3):
                progress = (t - 10) / 5
                ped_x = int(width//2 + (i-1)*50 + progress * 200)
                ped_y = height//2
                cv2.circle(frame, (ped_x, ped_y), 12, (0, 255, 255), -1)
                cv2.rectangle(frame, (ped_x-5, ped_y+5), (ped_x+5, ped_y+20), (0, 255, 255), -1)
            cv2.putText(frame, "🚶 PEDESTRIANS CROSSING", (width//2-150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Pedestrian test video created: {output_file}")
    return output_file


def create_comprehensive_demo_video(output_file="demo_traffic.mp4", duration=45, fps=30):
    """
    Create comprehensive demo video showing all scenarios
    """
    print(f"🎬 Creating comprehensive demo video: {output_file}")
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    # Define regions
    lane_region = [(width//10, height-150), (width//3, height-150), 
                   (width//3+30, height-120), (width//10, height-120)]
    crossing_region = [(width//3, height//2-50), (2*width//3, height//2-50),
                       (2*width//3, height//2+50), (width//3, height//2+50)]
    
    # Vehicle positions for smooth animation
    vehicle_positions = {
        'car1': -100,
        'car2': width + 100,
        'bus1': 0,
        'blocking1': None,
        'blocking2': None
    }
    
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        t = frame_num / fps
        
        # Sky
        cv2.rectangle(frame, (0, 0), (width, height-250), (135, 206, 235), -1)
        
        # Road
        cv2.rectangle(frame, (0, height-250), (width, height), (50, 50, 50), -1)
        
        # Lane markings
        for i in range(0, width, 50):
            cv2.rectangle(frame, (i, height-125), (i+30, height-123), (255, 255, 255), -1)
        
        # Free-left lane
        pts = np.array(lane_region, np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        cv2.putText(frame, "FREE LEFT LANE", (lane_region[0][0], lane_region[0][1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Zebra crossing
        x1, y1 = crossing_region[0]
        x2, y2 = crossing_region[2]
        stripe_width = (x2 - x1) // 8
        for i in range(8):
            x = x1 + i * stripe_width
            if i % 2 == 0:
                cv2.rectangle(frame, (x, y1), (x + stripe_width, y2), (255, 255, 255), -1)
        cv2.polylines(frame, [np.array(crossing_region, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame, "PEDESTRIAN CROSSING", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Moving vehicles
        # Car moving left to right
        vehicle_positions['car1'] += 5
        if vehicle_positions['car1'] > width + 100:
            vehicle_positions['car1'] = -100
        if 0 <= vehicle_positions['car1'] <= width - 80:
            cv2.rectangle(frame, (int(vehicle_positions['car1']), height-190), 
                         (int(vehicle_positions['car1'])+70, height-145), (0, 255, 0), -1)
            cv2.putText(frame, "Car", (int(vehicle_positions['car1'])+25, height-165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Oncoming car
        vehicle_positions['car2'] -= 4
        if vehicle_positions['car2'] < -100:
            vehicle_positions['car2'] = width + 100
        if 0 <= vehicle_positions['car2'] <= width - 80:
            cv2.rectangle(frame, (int(vehicle_positions['car2']), height-190), 
                         (int(vehicle_positions['car2'])+70, height-145), (255, 0, 0), -1)
            cv2.putText(frame, "Oncoming", (int(vehicle_positions['car2'])+15, height-165),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # SCENARIO 1: Normal flow (0-10s)
        if t < 10:
            cv2.putText(frame, "🟡 SCENARIO 1: Normal Flow - FREE LEFT MODE", (width//2-250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # SCENARIO 2: Single vehicle blocking (10-18s)
        elif t < 18:
            cv2.rectangle(frame, (200, height-145), (270, height-105), (0, 0, 255), -1)
            cv2.putText(frame, "BLOCKING!", (210, height-120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "⚠️ SINGLE VEHICLE BLOCKING", (width//2-200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # SCENARIO 3: Multiple blocking - Protected left triggered (18-25s)
        elif t < 25:
            cv2.rectangle(frame, (180, height-145), (250, height-105), (0, 0, 255), -1)
            cv2.rectangle(frame, (270, height-145), (340, height-105), (0, 0, 255), -1)
            cv2.putText(frame, "BLOCKING!", (190, height-120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "BLOCKING!", (280, height-120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "🟢 PROTECTED LEFT TRIGGERED - Multiple vehicles blocking", (width//2-300, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Show oncoming traffic stopped
            cv2.rectangle(frame, (int(vehicle_positions['car2']), height-190), 
                         (int(vehicle_positions['car2'])+70, height-145), (100, 100, 100), -1)
        
        # SCENARIO 4: Pedestrians waiting (25-32s)
        elif t < 32:
            for i in range(3):
                ped_x = width//2 + (i-1)*50
                ped_y = height//2 - 30
                cv2.circle(frame, (ped_x, ped_y), 12, (0, 0, 255), -1)
                cv2.rectangle(frame, (ped_x-5, ped_y+5), (ped_x+5, ped_y+20), (0, 0, 255), -1)
                cv2.putText(frame, f"P{i+1}", (ped_x-5, ped_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, "🚶 PEDESTRIAN MODE ACTIVATED - All vehicles STOP", (width//2-300, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Stop vehicles
            cv2.rectangle(frame, (int(vehicle_positions['car1']), height-190), 
                         (int(vehicle_positions['car1'])+70, height-145), (100, 100, 100), -1)
            cv2.rectangle(frame, (int(vehicle_positions['car2']), height-190), 
                         (int(vehicle_positions['car2'])+70, height-145), (100, 100, 100), -1)
        
        # SCENARIO 5: Return to normal (32-45s)
        else:
            cv2.putText(frame, "🟡 RETURN TO FREE LEFT MODE - System ready for next cycle", (width//2-300, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Signal status overlay
        if 18 <= t < 25:
            signal_text = "🟢 PROTECTED LEFT ACTIVE"
            signal_color = (0, 255, 0)
        elif 25 <= t < 32:
            signal_text = "🔴 PEDESTRIAN MODE - STOP"
            signal_color = (0, 0, 255)
        else:
            signal_text = "🟡 FREE LEFT MODE - YIELD"
            signal_color = (0, 255, 255)
        
        cv2.rectangle(frame, (width-250, 20), (width-20, 80), (0, 0, 0), -1)
        cv2.putText(frame, signal_text, (width-240, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, signal_color, 1)
        
        # Timestamp
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
        
        if frame_num % (fps * 5) == 0:
            print(f"  Progress: {frame_num/fps:.0f}/{duration}s")
    
    out.release()
    print(f"✅ Comprehensive demo video created: {output_file}")
    return output_file


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 TEST VIDEO GENERATOR FOR INTELLIGENT FREE LEFT TURN SYSTEM")
    print("="*60)
    
    # Create test videos
    create_test_video_with_traffic("test_traffic.mp4", duration=30)
    create_pedestrian_test_video("test_pedestrians.mp4", duration=20)
    create_comprehensive_demo_video("demo_traffic.mp4", duration=45)
    
    print("\n" + "="*60)
    print("✅ All test videos created successfully!")
    print("\n📁 Video files created:")
    print("  1. test_traffic.mp4 - Basic traffic with blocking vehicles")
    print("  2. test_pedestrians.mp4 - Pedestrian crossing scenarios")
    print("  3. demo_traffic.mp4 - Comprehensive demo with all scenarios")
    print("\n🚀 Next steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Select 'Real-time Monitoring' mode")
    print("  3. Upload one of the test videos")
    print("  4. Click START to begin dry run")
    print("="*60)