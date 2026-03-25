#!/usr/bin/env python3
"""
Dry Run Script - Automated Testing of the System
Runs the system with pre-recorded videos and generates test results
"""

import sys
import os
import time
import json
import cv2
import numpy as np
from datetime import datetime
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import UnifiedDetector
from core.signal_controller import EnhancedSignalController
from core.zebra_detector import EnhancedZebraDetector


class DryRunTester:
    """
    Automated dry run tester for the system
    """
    
    def __init__(self, video_path, output_dir="dry_run_results"):
        self.video_path = video_path
        self.output_dir = output_dir
        self.results = {
            'test_name': os.path.basename(video_path),
            'timestamp': datetime.now().isoformat(),
            'frames_processed': 0,
            'detections': [],
            'signal_changes': [],
            'blocking_events': [],
            'pedestrian_events': []
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.detector = UnifiedDetector(
            auto_detect_crossing=True,
            conf_threshold=0.4,
            enhance_visibility=True,
            debug_mode=True
        )
        self.controller = EnhancedSignalController()
        
        print(f"🎬 Dry Run Tester Initialized")
        print(f"   Video: {video_path}")
        print(f"   Output: {output_dir}")
    
    def run(self):
        """Run the dry test"""
        print("\n" + "="*60)
        print("🚦 STARTING DRY RUN")
        print("="*60)
        
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {self.video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\n📹 Video Info:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   Total Frames: {total_frames}")
        print(f"\n🔍 Processing frames...\n")
        
        frame_count = 0
        start_time = time.time()
        
        # Create video writer for output
        output_video = os.path.join(self.output_dir, "processed_video.mp4")
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, int(fps), 
                              (int(cap.get(3)), int(cap.get(4))))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect
            detections, annotated = self.detector.detect(frame)
            
            # Get lane vehicles and pedestrians
            lane_vehicles = [d for d in detections if hasattr(d, 'in_lane') and d.in_lane]
            pedestrians = [d for d in detections if hasattr(d, 'type') and 
                          hasattr(d.type, 'value') and d.type.value == 'pedestrian']
            
            # Update controller
            old_phase = self.controller.current_phase.value
            new_phase = self.controller.update_detections(lane_vehicles, pedestrians)
            
            # Log events
            if old_phase != new_phase.value:
                self.results['signal_changes'].append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'from': old_phase,
                    'to': new_phase.value
                })
                print(f"  Frame {frame_count}: {old_phase} → {new_phase.value}")
            
            # Log blocking events
            for v in lane_vehicles:
                if hasattr(v, 'is_stationary') and v.is_stationary and v.stationary_time > 3:
                    self.results['blocking_events'].append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'type': v.class_name,
                        'duration': v.stationary_time
                    })
            
            # Log pedestrian events
            for p in pedestrians:
                if hasattr(p, 'at_crossing') and p.at_crossing:
                    self.results['pedestrian_events'].append({
                        'frame': frame_count,
                        'time': frame_count / fps,
                        'confidence': p.confidence
                    })
            
            # Add overlay text
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated, f"Phase: {new_phase.value}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(annotated, f"Blocking: {len(lane_vehicles)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            out.write(annotated)
            
            # Progress update
            if frame_count % (fps * 5) == 0:
                progress = frame_count / total_frames * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        out.release()
        
        elapsed_time = time.time() - start_time
        self.results['frames_processed'] = frame_count
        self.results['processing_time'] = elapsed_time
        self.results['fps_processed'] = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Print summary
        self._print_summary()
        
        # Save results
        self._save_results()
        
        print(f"\n✅ Dry run completed!")
        print(f"   Processed video saved: {output_video}")
        print(f"   Results saved: {os.path.join(self.output_dir, 'results.json')}")
    
    def _print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 DRY RUN SUMMARY")
        print("="*60)
        
        print(f"\n📹 Video Processed:")
        print(f"   Frames: {self.results['frames_processed']}")
        print(f"   Processing Time: {self.results['processing_time']:.2f}s")
        print(f"   Processing Speed: {self.results['fps_processed']:.1f} fps")
        
        print(f"\n🚦 Signal Changes:")
        print(f"   Total Changes: {len(self.results['signal_changes'])}")
        for change in self.results['signal_changes'][:5]:
            print(f"   - {change['time']:.1f}s: {change['from']} → {change['to']}")
        
        print(f"\n🚗 Blocking Events:")
        print(f"   Total: {len(self.results['blocking_events'])}")
        for event in self.results['blocking_events'][:5]:
            print(f"   - {event['time']:.1f}s: {event['type']} blocked for {event['duration']:.1f}s")
        
        print(f"\n🚶 Pedestrian Events:")
        print(f"   Total: {len(self.results['pedestrian_events'])}")
        for event in self.results['pedestrian_events'][:5]:
            print(f"   - {event['time']:.1f}s: Pedestrian detected (conf: {event['confidence']:.2f})")
    
    def _save_results(self):
        """Save results to JSON"""
        output_file = os.path.join(self.output_dir, "results.json")
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📁 Results saved: {output_file}")


def run_all_tests():
    """Run tests on all sample videos"""
    print("\n" + "="*60)
    print("🚦 INTELLIGENT FREE LEFT TURN SYSTEM - DRY RUN SUITE")
    print("="*60)
    
    # Check if test videos exist, create if not
    test_videos = ["test_traffic.mp4", "test_pedestrians.mp4", "demo_traffic.mp4"]
    
    for video in test_videos:
        if not os.path.exists(video):
            print(f"\n⚠️ Test video {video} not found. Run create_test_videos.py first.")
            return
    
    # Run tests
    for video in test_videos:
        print(f"\n{'='*60}")
        print(f"📹 TESTING: {video}")
        print(f"{'='*60}")
        
        output_dir = f"dry_run_results/{video.replace('.mp4', '')}"
        tester = DryRunTester(video, output_dir)
        tester.run()
        
        time.sleep(2)
    
    print("\n" + "="*60)
    print("✅ ALL DRY RUN TESTS COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dry Run Tester')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_tests()
    elif args.video:
        output_dir = f"dry_run_results/{os.path.basename(args.video).replace('.mp4', '')}"
        tester = DryRunTester(args.video, output_dir)
        tester.run()
    else:
        print("Usage:")
        print("  python dry_run.py --all              # Run all tests")
        print("  python dry_run.py --video test.mp4   # Test specific video")
        print("\nFirst, create test videos:")
        print("  python create_test_videos.py")