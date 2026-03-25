#!/usr/bin/env python3
"""
Test script for enhanced zebra crossing detector
"""

import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.zebra_detector import EnhancedZebraDetector, ZebraCrossingTracker


def create_synthetic_zebra():
    """Create a synthetic zebra crossing pattern for testing"""
    width, height = 800, 600
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw road (dark gray)
    cv2.rectangle(img, (0, height-200), (width, height), (50, 50, 50), -1)
    
    # Draw zebra stripes
    stripe_width = 40
    stripe_height = 80
    num_stripes = 8
    start_x = (width - num_stripes * stripe_width) // 2
    
    for i in range(num_stripes):
        x = start_x + i * stripe_width
        color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
        cv2.rectangle(img, (x, height-140), (x + stripe_width, height-60), color, -1)
    
    return img


def test_on_image(image_path=None):
    """Test zebra detector on a single image"""
    print("\n" + "="*60)
    print("ZEBRA CROSSING DETECTOR TEST")
    print("="*60)
    
    if image_path and os.path.exists(image_path):
        frame = cv2.imread(image_path)
        print(f"📸 Testing on: {image_path}")
    else:
        print("📸 Creating synthetic test pattern...")
        frame = create_synthetic_zebra()
        cv2.imwrite("synthetic_zebra.jpg", frame)
        print("   Saved: synthetic_zebra.jpg")
    
    detector = EnhancedZebraDetector(min_stripes=3, confidence_threshold=0.5)
    tracker = ZebraCrossingTracker(stability_frames=3)
    
    # Test multiple frames to show stability
    results = []
    for i in range(10):
        # Add slight variation to simulate real video
        if i > 0 and frame is not None:
            test_frame = frame.copy()
            # Add slight random offset to simulate camera shake
            offset = np.random.randint(-2, 3, 2)
            M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]]) # pyright: ignore[reportArgumentType]
            test_frame = cv2.warpAffine(test_frame, M, (frame.shape[1], frame.shape[0])) # pyright: ignore[reportArgumentType, reportCallIssue]
        else:
            test_frame = frame
        
        result = detector.detect(test_frame) # pyright: ignore[reportArgumentType]
        tracked = tracker.update(result)
        results.append(tracked)
        
        if tracked:
            print(f"  Frame {i+1}: Detected (conf: {tracked.confidence:.2f})")
        else:
            print(f"  Frame {i+1}: No detection")
    
    # Show detection on original
    if results[-1]:
        annotated = detector.draw_detection(frame, results[-1]) # pyright: ignore[reportArgumentType]
        cv2.imshow("Zebra Crossing Detection", annotated)
        cv2.imwrite("detected_zebra.jpg", annotated)
        print("\n✅ Detection successful!")
        print(f"   Confidence: {results[-1].confidence:.2f}")
        print(f"   Orientation: {results[-1].orientation}")
        print(f"   Stripe count: {results[-1].stripe_count}")
        print("   Saved: detected_zebra.jpg")
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    else:
        print("\n❌ No zebra crossing detected")
        print("   Tips: Ensure good lighting and clear visibility")


def main():
    """Main test function"""
    if len(sys.argv) > 1:
        test_on_image(sys.argv[1])
    else:
        test_on_image()


if __name__ == "__main__":
    main()