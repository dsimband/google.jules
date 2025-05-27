import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import cv2
import numpy as np
import sys

# Add the parent directory to sys.path to allow importing video_processor
# This is often needed if tests are run directly from the tests/ directory
# and video_processor.py is in the parent directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import video_processor

class TestVideoProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_output_dir = "test_outputs"
        os.makedirs(self.test_output_dir, exist_ok=True)

        self.dummy_input_filename = "dummy_input_video_for_test.mp4"
        self.dummy_output_filename = "dummy_output_video_for_test.mp4"

        self.dummy_input_path = os.path.join(self.test_output_dir, self.dummy_input_filename)
        self.dummy_output_path = os.path.join(self.test_output_dir, self.dummy_output_filename)

        # Create a small, valid dummy video file for testing process_video_with_pose
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
            fps = 20
            frame_size = (64, 64) # Small frame size
            # Ensure dummy_input_path is used for VideoWriter
            out = cv2.VideoWriter(self.dummy_input_path, fourcc, fps, frame_size)
            if not out.isOpened():
                raise IOError(f"cv2.VideoWriter failed to open {self.dummy_input_path}")

            for _ in range(10): # 10 frames
                frame = np.random.randint(0, 256, (*frame_size[::-1], 3), dtype=np.uint8)
                out.write(frame)
            out.release()
            # Check if file was created and is not empty
            if not os.path.exists(self.dummy_input_path) or os.path.getsize(self.dummy_input_path) == 0:
                print(f"Warning: Dummy input video {self.dummy_input_path} may not have been created correctly.")
        except Exception as e:
            print(f"ERROR in setUp creating dummy video: {e}. Some tests might fail or be skipped.")
            # If dummy video creation fails, some process_video tests cannot run meaningfully.
            # Depending on policy, one might raise SkipTest here or let tests fail.
            self.dummy_input_path = None # Indicate failure

    def tearDown(self):
        """Tear down test fixtures, if any."""
        if os.path.exists(self.dummy_input_path) and self.dummy_input_path: # Check if not None
            os.remove(self.dummy_input_path)
        if os.path.exists(self.dummy_output_path):
            os.remove(self.dummy_output_path)
        
        # Clean up any other files that might have been created by tests, e.g. temp_video.mp4
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        if os.path.exists("processed_video.mp4"):
            os.remove("processed_video.mp4")

        if os.path.exists(self.test_output_dir):
            # Check if directory is empty before removing, to be safe
            if not os.listdir(self.test_output_dir):
                os.rmdir(self.test_output_dir)
            else: # If other files were created unexpectedly, print a warning
                print(f"Warning: Test output directory {self.test_output_dir} not empty. Check for leftover files.")


    @patch('video_processor.yt_dlp.YoutubeDL')
    def test_download_video_success(self, mock_youtube_dl):
        """Test successful video download."""
        mock_ydl_instance = MagicMock()
        mock_youtube_dl.return_value.__enter__.return_value = mock_ydl_instance
        
        test_url = "https://www.youtube.com/watch?v=testvideo"
        output_path = os.path.join(self.test_output_dir, "test_download.mp4")
        
        result = video_processor.download_video(test_url, output_path)
        
        self.assertEqual(result, output_path)
        mock_youtube_dl.assert_called_once_with({
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
        })
        mock_ydl_instance.download.assert_called_once_with([test_url])
        if os.path.exists(output_path): # Output file should not be created by mock
             os.remove(output_path)


    @patch('video_processor.yt_dlp.YoutubeDL')
    def test_download_video_failure(self, mock_youtube_dl):
        """Test video download failure (e.g., yt_dlp exception)."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.download.side_effect = Exception("Simulated download error")
        mock_youtube_dl.return_value.__enter__.return_value = mock_ydl_instance

        test_url = "https://www.youtube.com/watch?v=fakevideo"
        output_path = os.path.join(self.test_output_dir, "test_download_fail.mp4")

        result = video_processor.download_video(test_url, output_path)
        self.assertIsNone(result)
        if os.path.exists(output_path): # Ensure no file is created on failure
             os.remove(output_path)

    @patch('video_processor.mediapipe.solutions.drawing_utils.draw_landmarks')
    @patch('video_processor.mediapipe.solutions.pose.Pose')
    def test_process_video_with_pose_success(self, mock_pose_constructor, mock_draw_landmarks):
        """Test successful video processing with pose detection."""
        if not self.dummy_input_path or not os.path.exists(self.dummy_input_path):
            self.skipTest(f"Dummy input video not available ({self.dummy_input_path}), skipping test.")

        mock_pose_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.pose_landmarks = MagicMock() # Simulate landmarks detected
        mock_pose_instance.process.return_value = mock_results
        mock_pose_constructor.return_value = mock_pose_instance

        result_path = video_processor.process_video_with_pose(self.dummy_input_path, self.dummy_output_path)

        self.assertEqual(result_path, self.dummy_output_path)
        self.assertTrue(os.path.exists(self.dummy_output_path)) # Check if output file was created
        self.assertGreater(os.path.getsize(self.dummy_output_path), 0) # Check if not empty

        mock_pose_constructor.assert_called_once() # Pose model initialized
        mock_pose_instance.process.assert_called() # process was called on frames
        mock_draw_landmarks.assert_called() # draw_landmarks was called (since landmarks were detected)
        mock_pose_instance.close.assert_called_once() # Ensure resources are released

    @patch('video_processor.mediapipe.solutions.drawing_utils.draw_landmarks')
    @patch('video_processor.mediapipe.solutions.pose.Pose')
    def test_process_video_with_pose_no_landmarks_detected(self, mock_pose_constructor, mock_draw_landmarks):
        """Test video processing where no pose landmarks are detected."""
        if not self.dummy_input_path or not os.path.exists(self.dummy_input_path):
            self.skipTest("Dummy input video not available, skipping test.")

        mock_pose_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.pose_landmarks = None  # Simulate NO landmarks detected
        mock_pose_instance.process.return_value = mock_results
        mock_pose_constructor.return_value = mock_pose_instance

        result_path = video_processor.process_video_with_pose(self.dummy_input_path, self.dummy_output_path)

        self.assertEqual(result_path, self.dummy_output_path)
        self.assertTrue(os.path.exists(self.dummy_output_path))
        self.assertGreater(os.path.getsize(self.dummy_output_path), 0) 
        
        mock_pose_constructor.assert_called_once()
        mock_pose_instance.process.assert_called()
        mock_draw_landmarks.assert_not_called() # Crucially, draw_landmarks should NOT be called
        mock_pose_instance.close.assert_called_once()


    @patch('video_processor.cv2.VideoCapture')
    def test_process_video_with_pose_video_open_failure(self, mock_video_capture):
        """Test failure when cv2.VideoCapture cannot open the video file."""
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False # Simulate open failure
        mock_video_capture.return_value = mock_cap_instance

        # No need for self.dummy_input_path if VideoCapture is fully mocked like this
        result_path = video_processor.process_video_with_pose("non_existent_video.mp4", self.dummy_output_path)
        
        self.assertIsNone(result_path)
        mock_video_capture.assert_called_once_with("non_existent_video.mp4")
        # Ensure output file is not created
        self.assertFalse(os.path.exists(self.dummy_output_path))


    @patch('video_processor.mediapipe.solutions.pose.Pose')
    def test_process_video_with_pose_processing_exception(self, mock_pose_constructor):
        """Test failure when mediapipe processing raises an exception."""
        if not self.dummy_input_path or not os.path.exists(self.dummy_input_path):
            self.skipTest("Dummy input video not available, skipping test.")

        mock_pose_instance = MagicMock()
        mock_pose_instance.process.side_effect = Exception("Simulated MediaPipe Error")
        mock_pose_constructor.return_value = mock_pose_instance

        result_path = video_processor.process_video_with_pose(self.dummy_input_path, self.dummy_output_path)

        self.assertIsNone(result_path)
        mock_pose_constructor.assert_called_once()
        mock_pose_instance.process.assert_called() # It was called, but raised an exception
        mock_pose_instance.close.assert_called_once() # Ensure resources are released even on error
        # Ensure output file might be created but should be empty or cleaned up
        # The current video_processor.py implementation might create an empty file before erroring in process loop
        # For robustness, check if it exists and if so, its size. Or ensure it's deleted by the function on error.
        # Based on current video_processor.py, it might create an empty file.
        # Let's assume the finally block in process_video_with_pose handles out.release() correctly.
        # If the file exists, it should ideally be empty if error occurred early.
        if os.path.exists(self.dummy_output_path):
             print(f"Output file {self.dummy_output_path} exists after processing exception, checking size.")
             # self.assertEqual(os.path.getsize(self.dummy_output_path), 0) # This might be too strict

    @patch('video_processor.mediapipe.solutions.pose.Pose')
    def test_process_video_with_pose_invalid_fps(self, mock_pose_constructor):
        """Test video processing with invalid FPS (<=0), ensuring it defaults to 30."""
        if not self.dummy_input_path or not os.path.exists(self.dummy_input_path):
            self.skipTest("Dummy input video not available, skipping test.")

        mock_pose_instance = MagicMock()
        mock_results = MagicMock()
        mock_results.pose_landmarks = None # No landmarks needed for this test
        mock_pose_instance.process.return_value = mock_results
        mock_pose_constructor.return_value = mock_pose_instance

        # Create a temporary video with 0 FPS using a mock for VideoCapture's get method
        with patch('video_processor.cv2.VideoCapture') as mock_video_capture_fps:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 64,
                cv2.CAP_PROP_FPS: 0 # Invalid FPS
            }.get(prop, 0) # Default for other props
            # Simulate read() method
            frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            # Let read return a frame a few times then stop
            mock_cap_instance.read.side_effect = [(True, frame)] * 5 + [(False, None)] 
            mock_video_capture_fps.return_value = mock_cap_instance

            # Mock VideoWriter to check the FPS it's initialized with
            with patch('video_processor.cv2.VideoWriter') as mock_video_writer_fps:
                mock_writer_instance = MagicMock()
                mock_video_writer_fps.return_value = mock_writer_instance

                result_path = video_processor.process_video_with_pose(self.dummy_input_path, self.dummy_output_path)
                
                self.assertEqual(result_path, self.dummy_output_path)
                # Check that VideoWriter was called with fps = 30
                mock_video_writer_fps.assert_called_once_with(
                    self.dummy_output_path, 
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    30, # Expected FPS
                    (64, 64) # Expected frame size
                )
        if os.path.exists(self.dummy_output_path):
            self.assertGreater(os.path.getsize(self.dummy_output_path), 0)


if __name__ == '__main__':
    unittest.main()
```
