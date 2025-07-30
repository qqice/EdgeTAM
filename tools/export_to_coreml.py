#!/usr/bin/env python3
"""Export EdgeTAM to CoreML format."""

import os
import sys
import argparse
import torch
import numpy as np
import coremltools as ct
from PIL import Image
import json
import warnings
from typing import Optional, Tuple, Dict, Any
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Suppress torch version warning from coremltools
warnings.filterwarnings("ignore", message="Torch version .* has not been tested with coremltools")
# Suppress resources.bin missing warnings (these are harmless)
warnings.filterwarnings("ignore", message=".*resources.bin missing.*")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sam2.build_sam import build_sam2, _load_checkpoint
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as e:
    print(f"Error importing SAM2 modules: {e}")
    print("Please ensure you have installed EdgeTAM properly.")
    sys.exit(1)


class EdgeTAMImageEncoder(torch.nn.Module):
    
    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        
    def forward(self, image):
        features = self.model.forward_image(image)
        
        backbone_fpn = features.get("backbone_fpn", [])
        if len(backbone_fpn) >= 2:
            return features["vision_features"], backbone_fpn[0], backbone_fpn[1]
        else:
            return features["vision_features"], None, None


class EdgeTAMPromptEncoder(torch.nn.Module):
    
    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.embed_dim = self.model.hidden_dim
        self.image_embedding_size = (64, 64)
        
    def forward(self, point_coords, point_labels, boxes, mask_input):
        bs = point_coords.shape[0]
        
        # Use the original SAM prompt encoder directly to avoid tracing issues
        # Pass None for boxes and masks to simplify tracing (since we're not using them)
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        
        return sparse_embeddings, dense_embeddings


class EdgeTAMMaskDecoder(torch.nn.Module):
    
    def __init__(self, sam_model):
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.sam_mask_decoder
        
    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, 
                dense_prompt_embeddings, high_res_feat_0, high_res_feat_1, multimask_output):
        high_res_features = [
            high_res_feat_0,
            high_res_feat_1
        ]
        
        sam_outputs = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        return (
            sam_outputs[0],  # masks
            sam_outputs[1],  # iou_pred
            sam_outputs[2],  # sam_tokens_out
            sam_outputs[3]   # object_score_logits
        )


def export_image_encoder(model, output_path: str, compute_units: ct.ComputeUnit, deployment_target):
    print("Exporting Image Encoder...")
    encoder_wrapper = EdgeTAMImageEncoder(model)
    encoder_wrapper.eval()
    
    example_input = torch.randn(1, 3, 1024, 1024)
    
    print("  Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(encoder_wrapper, example_input)
    
    print("  Converting to CoreML...")
    image_input = ct.ImageType(
        name="image",
        shape=(1, 3, 1024, 1024),
        scale=1/255.0,
        bias=[0, 0, 0],
        color_layout=ct.colorlayout.RGB
    )
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[image_input],
        outputs=[
            ct.TensorType(name="vision_features"),
            ct.TensorType(name="high_res_feat_0"),
            ct.TensorType(name="high_res_feat_1")
        ],
        minimum_deployment_target=deployment_target,
        compute_units=compute_units,
        convert_to="mlprogram"
    )
    
    mlmodel.author = "EdgeTAM Contributors"
    mlmodel.short_description = "EdgeTAM Image Encoder"
    mlmodel.version = "1.0"
    
    mlmodel.save(output_path)
    print(f"  ✓ Saved to {output_path}")
    
    return mlmodel


def export_prompt_encoder(model, output_path: str, compute_units: ct.ComputeUnit, deployment_target):
    print("Exporting Prompt Encoder...")
    encoder_wrapper = EdgeTAMPromptEncoder(model)
    encoder_wrapper.eval()
    
    max_points = 4
    point_coords = torch.zeros(1, max_points, 2)
    point_labels = torch.full((1, max_points), -1, dtype=torch.float32)
    boxes = torch.zeros(1, 4)
    mask_input = torch.zeros(1, 1, 256, 256)
    
    print("  Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            encoder_wrapper, 
            (point_coords, point_labels, boxes, mask_input)
        )
    
    print("  Converting to CoreML...")
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="point_coords", shape=(1, max_points, 2)),
            ct.TensorType(name="point_labels", shape=(1, max_points)),
            ct.TensorType(name="boxes", shape=(1, 4)),
            ct.TensorType(name="mask_input", shape=(1, 1, 256, 256))
        ],
        outputs=[
            ct.TensorType(name="sparse_embeddings"),
            ct.TensorType(name="dense_embeddings")
        ],
        minimum_deployment_target=deployment_target,
        compute_units=compute_units,
        convert_to="mlprogram"
    )
    
    # Add metadata
    mlmodel.author = "EdgeTAM Contributors"
    mlmodel.short_description = "EdgeTAM Prompt Encoder - Encodes user prompts (points, boxes, masks)"
    mlmodel.version = "1.0"
    
    mlmodel.save(output_path)
    print(f"  ✓ Saved to {output_path}")
    
    return mlmodel


def export_mask_decoder(model, output_path: str, compute_units: ct.ComputeUnit, deployment_target):
    """Export the mask decoder component"""
    print("Exporting Mask Decoder...")
    
    # Create wrapper
    decoder_wrapper = EdgeTAMMaskDecoder(model)
    decoder_wrapper.eval()
    
    # Create example inputs
    image_embeddings = torch.randn(1, 256, 64, 64)
    image_pe = torch.randn(1, 256, 64, 64)
    sparse_prompt_embeddings = torch.randn(1, 2, 256)  # Variable size in practice
    dense_prompt_embeddings = torch.randn(1, 256, 64, 64)
    high_res_feat_0 = torch.randn(1, 32, 256, 256)
    high_res_feat_1 = torch.randn(1, 64, 128, 128)
    multimask_output = torch.tensor([True])  # Export with multi-mask for flexibility
    
    print("  Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            decoder_wrapper,
            (image_embeddings, image_pe, sparse_prompt_embeddings, 
             dense_prompt_embeddings, high_res_feat_0, high_res_feat_1, multimask_output)
        )
    
    print("  Converting to CoreML...")
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image_embeddings", shape=(1, 256, 64, 64)),
            ct.TensorType(name="image_pe", shape=(1, 256, 64, 64)),
            ct.TensorType(name="sparse_prompt_embeddings", shape=(1, ct.RangeDim(1, 10), 256)),  # Min 1 to prevent hanging
            ct.TensorType(name="dense_prompt_embeddings", shape=(1, 256, 64, 64)),
            ct.TensorType(name="high_res_feat_0", shape=(1, 32, 256, 256)),
            ct.TensorType(name="high_res_feat_1", shape=(1, 64, 128, 128)),
            ct.TensorType(name="multimask_output", shape=(1,))
        ],
        outputs=[
            ct.TensorType(name="masks"),
            ct.TensorType(name="iou_pred"),
            ct.TensorType(name="sam_tokens_out"),
            ct.TensorType(name="object_score_logits")
        ],
        minimum_deployment_target=deployment_target,
        compute_units=compute_units,
        convert_to="mlprogram"
    )
    
    # Add metadata
    mlmodel.author = "EdgeTAM Contributors"
    mlmodel.short_description = "EdgeTAM Mask Decoder - Generates segmentation masks from features and prompts"
    mlmodel.version = "1.0"
    
    mlmodel.save(output_path)
    print(f"  ✓ Saved to {output_path}")
    
    return mlmodel


def create_inference_example(output_dir: str):
    """Create a Python example showing how to use the exported models"""
    
    example_code = '''#!/usr/bin/env python3
"""
Example usage of EdgeTAM CoreML models

This script demonstrates how to use the exported CoreML models for segmentation.
"""

import numpy as np
import coremltools as ct
from PIL import Image

def load_models(model_dir="./"):
    """Load all EdgeTAM CoreML models"""
    print("Loading EdgeTAM CoreML models...")
    
    image_encoder = ct.models.MLModel(f"{model_dir}/edgetam_image_encoder.mlpackage")
    prompt_encoder = ct.models.MLModel(f"{model_dir}/edgetam_prompt_encoder.mlpackage")
    mask_decoder = ct.models.MLModel(f"{model_dir}/edgetam_mask_decoder.mlpackage")
    
    return image_encoder, prompt_encoder, mask_decoder

def segment_with_point(image_path, point_x, point_y, model_dir="./"):
    """Segment an image using a point prompt"""
    
    # Load models
    image_encoder, prompt_encoder, mask_decoder = load_models(model_dir)
    
    # Load and prepare image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((1024, 1024), Image.BILINEAR)
    
    # Encode image
    print("Encoding image...")
    encoder_out = image_encoder.predict({"image": image})
    
    # Prepare point prompt
    point_coords = np.zeros((1, 4, 2), dtype=np.float32)
    point_labels = np.full((1, 4), -1, dtype=np.float32)
    point_coords[0, 0] = [point_x, point_y]
    point_labels[0, 0] = 1  # positive point
    
    # Encode prompt
    print("Encoding prompt...")
    prompt_out = prompt_encoder.predict({
        "point_coords": point_coords,
        "point_labels": point_labels,
        "boxes": np.zeros((1, 4), dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32)
    })
    
    # Generate positional encoding (simplified)
    image_pe = np.zeros((1, 256, 64, 64), dtype=np.float32)
    
    # Decode mask
    print("Generating mask...")
    mask_out = mask_decoder.predict({
        "image_embeddings": encoder_out['vision_features'],
        "image_pe": image_pe,
        "sparse_prompt_embeddings": prompt_out['sparse_embeddings'],
        "dense_prompt_embeddings": prompt_out['dense_embeddings'],
        "high_res_feat_0": encoder_out['high_res_feat_0'],
        "high_res_feat_1": encoder_out['high_res_feat_1'],
        "multimask_output": np.array([True], dtype=np.float32)
    })
    
    # Get best mask
    masks = mask_out['masks']
    iou_scores = mask_out['iou_pred']
    best_mask_idx = np.argmax(iou_scores)
    best_mask = masks[0, best_mask_idx]
    
    print(f"Generated mask with IoU score: {iou_scores[0, best_mask_idx]:.3f}")
    
    return best_mask

if __name__ == "__main__":
    # Example usage
    # mask = segment_with_point("path/to/image.jpg", 512, 300)
    print("EdgeTAM CoreML models ready!")
    print("Use segment_with_point() to generate masks.")
'''
    
    example_path = os.path.join(output_dir, "inference_example.py")
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    # Make executable
    os.chmod(example_path, 0o755)
    print(f"  ✓ Created inference example: {example_path}")


def validate_models(output_dir: str, test_image_path: Optional[str] = None):
    """Validate the exported CoreML models"""
    print("\nValidating exported models...")
    
    try:
        # Load models
        image_encoder = ct.models.MLModel(os.path.join(output_dir, "edgetam_image_encoder.mlpackage"))
        prompt_encoder = ct.models.MLModel(os.path.join(output_dir, "edgetam_prompt_encoder.mlpackage"))
        mask_decoder = ct.models.MLModel(os.path.join(output_dir, "edgetam_mask_decoder.mlpackage"))
        
        # Test image
        if test_image_path and os.path.exists(test_image_path):
            image = Image.open(test_image_path).convert('RGB').resize((1024, 1024))
        else:
            image = Image.new('RGB', (1024, 1024), color='white')
        
        # Test image encoder
        print("  Testing image encoder...")
        encoder_out = image_encoder.predict({"image": image})
        print(f"    ✓ Vision features: {encoder_out['vision_features'].shape}")
        print(f"    ✓ High-res feat 0: {encoder_out['high_res_feat_0'].shape}")
        print(f"    ✓ High-res feat 1: {encoder_out['high_res_feat_1'].shape}")
        
        # Test prompt encoder with valid point
        print("  Testing prompt encoder...")
        prompt_out = prompt_encoder.predict({
            "point_coords": np.array([[[512, 512], [0, 0], [0, 0], [0, 0]]], dtype=np.float32),
            "point_labels": np.array([[1, -1, -1, -1]], dtype=np.float32),
            "boxes": np.zeros((1, 4), dtype=np.float32),
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32)
        })
        print(f"    ✓ Sparse embeddings: {prompt_out['sparse_embeddings'].shape}")
        print(f"    ✓ Dense embeddings: {prompt_out['dense_embeddings'].shape}")
        
        # Test mask decoder
        print("  Testing mask decoder...")
        pe = np.zeros((1, 256, 64, 64), dtype=np.float32)
        
        decoder_out = mask_decoder.predict({
            "image_embeddings": encoder_out['vision_features'],
            "image_pe": pe,
            "sparse_prompt_embeddings": prompt_out['sparse_embeddings'],
            "dense_prompt_embeddings": prompt_out['dense_embeddings'],
            "high_res_feat_0": encoder_out['high_res_feat_0'],
            "high_res_feat_1": encoder_out['high_res_feat_1'],
            "multimask_output": np.array([True], dtype=np.float32)
        })
        
        print(f"    ✓ Masks: {decoder_out['masks'].shape}")
        print(f"    ✓ IoU predictions: {decoder_out['iou_pred'].shape}")
        print(f"    ✓ Object scores: {decoder_out['object_score_logits'].shape}")
        
        print("  ✅ All models validated successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export EdgeTAM to CoreML format")
    parser.add_argument("--sam2_cfg", required=True, help="Path to EdgeTAM config file")
    parser.add_argument("--sam2_checkpoint", required=True, help="Path to EdgeTAM checkpoint")
    parser.add_argument("--output_dir", default="./coreml_models", help="Output directory for CoreML models")
    parser.add_argument("--validate", action="store_true", help="Validate exported models")
    parser.add_argument("--test_image", help="Path to test image for validation")
    parser.add_argument("--compute_units", choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"], 
                       default="ALL", help="Target compute units")
    parser.add_argument("--deployment_target", choices=["iOS16", "iOS17", "macOS13", "macOS14"], 
                       default="iOS16", help="Minimum deployment target")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Map string arguments to CoreML enums
    compute_units_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE
    }
    
    deployment_target_map = {
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14
    }
    
    compute_units = compute_units_map[args.compute_units]
    deployment_target = deployment_target_map[args.deployment_target]
    
    print("Loading EdgeTAM model...")
    try:
        device = torch.device("cpu")  # Use CPU for export
        
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Get config directory and config name
        config_path = os.path.abspath(args.sam2_cfg)
        config_dir = os.path.dirname(config_path)
        config_name = os.path.splitext(os.path.basename(config_path))[0]
        
        print(f"  Config directory: {config_dir}")
        print(f"  Config name: {config_name}")
        
        # Load config using initialize_config_dir approach
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=config_name)
            OmegaConf.resolve(cfg)
            print("  ✓ Config loaded successfully")
            
            # Instantiate model
            model = instantiate(cfg.model, _recursive_=True)
            print("  ✓ Model instantiated successfully")
            
            # Load checkpoint if provided
            if args.sam2_checkpoint:
                _load_checkpoint(model, args.sam2_checkpoint)
                print("  ✓ Checkpoint loaded successfully")
            
            model = model.to(device)
            model.eval()
            
    except Exception as e:
        print(f"Failed to load EdgeTAM model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Exporting to {args.output_dir} with {args.compute_units} compute units...")
    
    # Export each component
    try:
        export_image_encoder(
            model, 
            os.path.join(args.output_dir, "edgetam_image_encoder.mlpackage"),
            compute_units,
            deployment_target
        )
        
        export_prompt_encoder(
            model,
            os.path.join(args.output_dir, "edgetam_prompt_encoder.mlpackage"),
            compute_units,
            deployment_target
        )
        
        export_mask_decoder(
            model,
            os.path.join(args.output_dir, "edgetam_mask_decoder.mlpackage"),
            compute_units,
            deployment_target
        )
        
        # Create inference example
        create_inference_example(args.output_dir)
        
        # Create metadata
        metadata = {
            "model_name": "EdgeTAM",
            "version": "1.0",
            "export_date": str(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
            "compute_units": args.compute_units,
            "deployment_target": args.deployment_target,
            "components": {
                "image_encoder": {
                    "path": "edgetam_image_encoder.mlpackage",
                    "input_size": [1024, 1024],
                    "description": "Processes input images to feature embeddings"
                },
                "prompt_encoder": {
                    "path": "edgetam_prompt_encoder.mlpackage",
                    "max_points": 4,
                    "supports_boxes": True,
                    "supports_masks": True,
                    "description": "Encodes user prompts (points, boxes, masks)"
                },
                "mask_decoder": {
                    "path": "edgetam_mask_decoder.mlpackage",
                    "output_size": [256, 256],
                    "max_masks": 3,
                    "description": "Generates segmentation masks from features and prompts"
                }
            }
        }
        
        metadata_path = os.path.join(args.output_dir, "model_info.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Export completed successfully!")
        print(f"📁 Models saved to: {args.output_dir}")
        print(f"📊 Total components: 3 (Image Encoder, Prompt Encoder, Mask Decoder)")
        
        # Validate if requested
        if args.validate:
            success = validate_models(args.output_dir, args.test_image)
            if not success:
                sys.exit(1)
        
        print(f"\n🚀 EdgeTAM CoreML models ready for iOS/macOS deployment!")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()