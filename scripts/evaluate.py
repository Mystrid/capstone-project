import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from egasn.models.egasn import EGASN
from egasn.data.cityscapes import CityscapesVideoDataset
from egasn.utils.metrics import Evaluator
from egasn.graph.epipolar import EpipolarGraphConstructor

CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EGASN on Cityscapes")
    parser.add_argument("--config", type=str, default="configs/egasn_cityscapes.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    val_dataset = CityscapesVideoDataset(split='val', transforms=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    model = EGASN(num_classes=19).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    evaluator = Evaluator(num_classes=19)
    graph_constructor = EpipolarGraphConstructor()
    
    total_uncertainty = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            frame_prev_np = batch['frame_prev_np']
            frame_curr_np = batch['frame_curr_np']
            frame_curr = batch['frame_curr'].to(device)
            label = batch['label'].to(device)
            
            b, c, h, w = frame_curr.shape
            x_prev_features = torch.zeros((b, 256, h // 4, w // 4), device=device)
            
            flow, _ = graph_constructor.construct(frame_prev_np[0], frame_curr_np[0])
            flow_batch = flow.unsqueeze(0).to(device)
            
            output = model(frame_curr, x_prev_features, flow_batch)
            preds = output['probs'].argmax(dim=1)
            uncertainty = output['uncertainty'].mean().item()
            
            evaluator.add_batch(label, preds)
            total_uncertainty += uncertainty
            num_samples += 1
            
    miou = evaluator.Mean_Intersection_over_Union()
    fwiou = evaluator.Frequency_Weighted_Intersection_over_Union()
    class_ious = evaluator.Intersection_over_Union()
    mean_uncertainty = total_uncertainty / num_samples if num_samples > 0 else 0
    
    print("-" * 40)
    print("EVALUATION RESULTS")
    print("-" * 40)
    print(f"{'Class':<20} | {'IoU':<10}")
    print("-" * 40)
    for cls_name, iou in zip(CITYSCAPES_CLASSES, class_ious):
        print(f"{cls_name:<20} | {iou:.4f}")
    print("-" * 40)
    print(f"mIoU:             {miou:.4f}")
    print(f"fwIoU:            {fwiou:.4f}")
    print(f"Mean Uncertainty: {mean_uncertainty:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
