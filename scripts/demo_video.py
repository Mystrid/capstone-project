import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

from egasn.models.egasn import EGASN
from egasn.graph.epipolar import EpipolarGraphConstructor

def parse_args():
    parser = argparse.ArgumentParser(description="Demo EGASN on a video file")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default="none", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/egasn_cityscapes.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="demo_output.mp4", help="Path to output video file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def apply_color_map(mask, num_classes=19):
    """Simple color map for visualization"""
    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    colors[0] = [128, 64, 128] # road
    colored_mask = colors[mask]
    return colored_mask.astype(np.uint8)

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    model = EGASN(num_classes=19).to(device)
    if args.checkpoint != "none":
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
    model.eval()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video {args.video}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Fix sizes to multiple of 32 for MobileNetV2
    width = (width // 32) * 32
    height = (height // 32) * 32
    
    graph_constructor = EpipolarGraphConstructor(output_size=(height // 32, width // 32))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height * 2))
    
    prev_frame = None
    prev_features = None
    
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            
            if prev_frame is None:
                prev_frame = frame_rgb
                # Dummy prev features for first frame
                b, c, h, w = frame_tensor.shape
                prev_features = torch.zeros((b, 256, h // 32, w // 32), device=device)
                
            flow, _ = graph_constructor.construct(prev_frame, frame_rgb)
            flow_tensor = flow.unsqueeze(0).to(device)
            
            output = model(frame_tensor, prev_features, flow_tensor)
            
            preds = output['probs'].argmax(dim=1).squeeze().cpu().numpy()
            uncertainty = output['uncertainty'].squeeze().cpu().numpy()
            prev_features = output['features']
            
            seg_viz = apply_color_map(preds)
            seg_viz = cv2.addWeighted(frame, 0.5, seg_viz, 0.5, 0)
            
            uncert_viz = (uncertainty / (uncertainty.max() + 1e-5) * 255).astype(np.uint8)
            uncert_viz = cv2.applyColorMap(uncert_viz, cv2.COLORMAP_JET)
            
            combined = np.vstack((seg_viz, uncert_viz))
            out.write(combined)
            
            prev_frame = frame_rgb
            pbar.update(1)
            
    cap.release()
    out.release()
    pbar.close()

if __name__ == "__main__":
    main()
