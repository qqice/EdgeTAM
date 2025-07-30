## EdgeTAM toolkits

This directory provides toolkits for additional EdgeTAM use cases.

### Semi-supervised VOS inference

The `vos_inference.py` script can be used to generate predictions for semi-supervised video object segmentation (VOS) evaluation on datasets such as [DAVIS](https://davischallenge.org/index.html), [MOSE](https://henghuiding.github.io/MOSE/) or the SA-V dataset.

After installing SAM 2 and its dependencies, it can be used as follows ([DAVIS 2017 dataset](https://davischallenge.org/davis2017/code.html) as an example). This script saves the prediction PNG files to the `--output_mask_dir`.
```bash
python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path-to-davis-2017/JPEGImages/480p \
  --input_mask_dir /path-to-davis-2017/Annotations/480p \
  --video_list_file /path-to-davis-2017/ImageSets/2017/val.txt \
  --output_mask_dir ./outputs/davis_2017_pred_pngs
```
(replace `/path-to-davis-2017` with the path to DAVIS 2017 dataset)

To evaluate on the SA-V dataset with per-object PNG files for the object masks, we need to **add the `--per_obj_png_file` flag** as follows (using SA-V val as an example). This script will also save per-object PNG files for the output masks under the `--per_obj_png_file` flag.
```bash
python ./tools/vos_inference.py \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_b+.yaml \
  --sam2_checkpoint ./checkpoints/sam2.1_hiera_base_plus.pt \
  --base_video_dir /path-to-sav-val/JPEGImages_24fps \
  --input_mask_dir /path-to-sav-val/Annotations_6fps \
  --video_list_file /path-to-sav-val/sav_val.txt \
  --per_obj_png_file \
  --output_mask_dir ./outputs/sav_val_pred_pngs
```
(replace `/path-to-sav-val` with the path to SA-V val)

Then, we can use the evaluation tools or servers for each dataset to get the performance of the prediction PNG files above.

Note: by default, the `vos_inference.py` script above assumes that all objects to track already appear on frame 0 in each video (as is the case in DAVIS, MOSE or SA-V). **For VOS datasets that don't have all objects to track appearing in the first frame (such as LVOS or YouTube-VOS), please add the `--track_object_appearing_later_in_video` flag when using `vos_inference.py`**.

### CoreML Export for iOS/macOS Deployment

Two scripts are available for exporting EdgeTAM models to CoreML format:

#### 1. `export_to_coreml.py` (Recommended)
Standard export using `torch.jit.trace`. Most reliable approach.

```bash
# Install CoreML dependencies
pip install -e ".[coreml]"

# Export EdgeTAM to CoreML
python ./tools/export_to_coreml.py \
  --sam2_cfg sam2/configs/edgetam.yaml \
  --sam2_checkpoint ./checkpoints/edgetam.pt \
  --output_dir ./coreml_models \
  --validate
```

#### 2. `export_to_coreml_hybrid.py` (Experimental)
Hybrid script/trace approach that attempts to better handle dynamic behavior.

```bash
python ./tools/export_to_coreml_hybrid.py \
  --sam2_cfg sam2/configs/edgetam.yaml \
  --sam2_checkpoint ./checkpoints/edgetam.pt \
  --output_dir ./coreml_models_hybrid \
  --validate
```

Both scripts create three CoreML models optimized for on-device inference:
- `edgetam_image_encoder.mlpackage` - Processes input images to feature embeddings
- `edgetam_prompt_encoder.mlpackage` - Handles user prompts (points, boxes, masks)
- `edgetam_mask_decoder.mlpackage` - Generates segmentation masks from features and prompts

**Important Notes:**
- Use EdgeTAM config (`sam2/configs/edgetam.yaml`) - SAM 2.1 configs fail due to unsupported bicubic interpolation
- Both scripts produce identical output structure
- The hybrid approach falls back to tracing due to model architecture constraints

Optional arguments:
- `--compute_units {ALL,CPU_ONLY,CPU_AND_GPU,CPU_AND_NE}` - Target compute units (default: ALL)
- `--deployment_target {iOS16,iOS17,macOS13,macOS14}` - Minimum deployment target (default: iOS16)
- `--test_image path/to/image.jpg` - Test image for validation

The exported models are ready for integration into iOS/macOS applications and support hardware acceleration via Neural Engine and GPU.
