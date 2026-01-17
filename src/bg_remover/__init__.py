import argparse
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rembg import remove, new_session
from tqdm import tqdm


def remove_background_from_video(
    input_path: str,
    output_path: str,
    use_gpu: bool = True,
    model_name: str = "u2net",
    output_frames: bool = False,
    output_webm: bool = False,
) -> None:
    """
    Remove background from an MP4 video and output with transparent background.
    
    Args:
        input_path: Path to input MP4 file
        output_path: Path to output video file or directory for frames
        use_gpu: Whether to use GPU acceleration (default: True)
        model_name: rembg model to use (u2net, u2netp, u2net_human_seg, etc.)
        output_frames: If True, output individual PNG frames instead of video (default: False)
        output_webm: If True, output WebM with transparency using FFmpeg (default: False)
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Check GPU availability
    if use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA not available. Using CPU instead.")
        use_gpu = False
    
    device = "cuda" if use_gpu else "cpu"
    print(f"Using device: {device}")
    
    # Initialize rembg session with specified model
    print(f"Loading model: {model_name}")
    session = new_session(model_name)
    
    # Open video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create output directory if needed
    if output_frames or output_webm:
        # For frames or WebM, we need a temporary directory for PNG frames
        if output_frames:
            frames_dir = output_path
            os.makedirs(frames_dir, exist_ok=True)
            print(f"Output mode: Individual PNG frames (with transparency)")
            print(f"Frames will be saved to: {frames_dir}/")
        else:
            # WebM mode - create temporary directory
            import tempfile
            frames_dir = tempfile.mkdtemp(prefix="bg_remover_")
            print(f"Output mode: WebM with transparency (using FFmpeg)")
            print(f"Temporary frames directory: {frames_dir}/")
        out = None
    else:
        # Output as video file with green screen
        frames_dir = None
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"Output mode: Video file with green screen")
        print(f"Note: Compositing transparent areas with green screen (#00FF00)")
        print(f"      Use chroma key in video editor to restore transparency")
        out = None
    
    # Process frames
    print("Processing frames...")
    with tqdm(total=total_frames, desc="Removing background", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Remove background
            output_image = remove(pil_image, session=session)
            
            # Convert back to OpenCV format (RGBA)
            output_array = np.array(output_image)
            
            if output_frames or output_webm:
                # Save individual PNG frames with transparency
                frame_filename = os.path.join(frames_dir, f"frame_{pbar.n:05d}.png")
                cv2.imwrite(frame_filename, cv2.cvtColor(output_array, cv2.COLOR_RGBA2BGRA))
            else:
                # Composite with green screen for video output
                if output_array.shape[2] == 4:  # RGBA
                    alpha = output_array[:, :, 3:4] / 255.0
                    rgb = output_array[:, :, :3]
                    # Green screen background (0, 255, 0) in RGB
                    green_bg = np.array([0, 255, 0]).reshape(1, 1, 3)
                    
                    # Composite: foreground * alpha + background * (1 - alpha)
                    composited = (rgb * alpha + green_bg * (1 - alpha)).astype(np.uint8)
                    frame_output = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
                else:
                    frame_output = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)
                
                # Store frames for video encoding
                if pbar.n == 0:
                    frames_buffer = []
                frames_buffer.append(frame_output)
            
            pbar.update(1)
    
    # Clean up and encode video if needed
    cap.release()
    
    if output_webm:
        # Encode WebM with transparency using FFmpeg
        print("Encoding WebM with transparency using FFmpeg...")
        
        # Ensure output path has .webm extension
        if not output_path.endswith('.webm'):
            output_path = os.path.splitext(output_path)[0] + '.webm'
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%05d.png'),
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',
            '-auto-alt-ref', '0',
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ Background removed successfully!")
            print(f"  WebM with transparency saved to: {output_path}")
            print(f"  File supports full alpha channel transparency!")
        except subprocess.CalledProcessError as e:
            print(f"Error encoding WebM: {e.stderr}")
            raise RuntimeError(f"FFmpeg encoding failed. Make sure FFmpeg is installed.")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg to use --webm option.")
        finally:
            # Clean up temporary frames directory
            import shutil
            shutil.rmtree(frames_dir, ignore_errors=True)
            print(f"  Temporary frames cleaned up")
    
    elif output_frames:
        print(f"✓ Background removed successfully!")
        print(f"  {total_frames} PNG frames saved to: {output_path}/")
        print(f"  Use a video editor or FFmpeg to create video from frames")
    
    else:
        # Encode video using OpenCV (with green screen background)
        output_ext = os.path.splitext(output_path)[1].lower()
        
        # Try different codecs based on format
        if output_ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif output_ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to create output video file: {output_path}")
        
        # Write all frames
        for frame in frames_buffer:
            out.write(frame)
        
        out.release()
        
        print(f"✓ Background removed successfully!")
        print(f"  Output saved to: {output_path}")
        print(f"  Background replaced with green screen (#00FF00)")
        print(f"  Use chroma key in your video editor to remove green and restore transparency")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from MP4 videos using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove background from video (output with transparent background)
  bg-remover input.mp4 output.webm
  
  # Use CPU instead of GPU
  bg-remover input.mp4 output.webm --no-gpu
  
  # Use a different model
  bg-remover input.mp4 output.webm --model u2net_human_seg
  
  # Output as MOV with transparency
  bg-remover input.mp4 output.mov
        """,
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input MP4 video file path",
    )
    
    parser.add_argument(
        "output",
        type=str,
        help="Output video file path (use .webm or .mov for transparency support)",
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (use CPU only)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="u2net",
        choices=["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"],
        help="Background removal model to use (default: u2net)",
    )
    
    parser.add_argument(
        "--frames",
        action="store_true",
        help="Output individual PNG frames instead of video (preserves full transparency)",
    )
    
    parser.add_argument(
        "--webm",
        action="store_true",
        help="Output WebM video with transparency using FFmpeg (requires FFmpeg installed)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.frames and args.webm:
        parser.error("Cannot use --frames and --webm together. Choose one output mode.")
    
    try:
        remove_background_from_video(
            input_path=args.input,
            output_path=args.output,
            use_gpu=not args.no_gpu,
            model_name=args.model,
            output_frames=args.frames,
            output_webm=args.webm,
        )
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
