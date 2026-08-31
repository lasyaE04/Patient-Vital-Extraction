# Patient Vital Extraction from Images of ICU Patient Monitors

Reference implementation for the three screen-localization and vital-sign
extraction pipelines compared in the paper *Patient Vital Extraction from Images
of ICU Patient Monitors* (CMBEC48/APIBQ 2026). Each pipeline extracts Heart Rate
(HR) and Oxygen Saturation (SpO2) from a photograph of a patient monitor.

| Notebook | Pipeline | Screen localization | Vital extraction |
| --- | --- | --- | --- |
| `SAM_HSV_OCR.ipynb` | 1 | Zero-shot Segment Anything + geometric priors | HSV colour bands + EasyOCR/Tesseract |
| `YOLO_HSV_OCR.ipynb` | 2 | Fine-tuned YOLOv8-seg | HSV colour bands + EasyOCR/Tesseract |
| `MOLMO2_4B.ipynb` | 3 | Fine-tuned YOLOv8-seg | Molmo2-4B vision-language model |

## Data layout

The notebooks are written for Google Colab and expect the dataset on Drive at
`DATASET_DIR` (set at the top of each notebook):

```
data/
  data.yaml
  train/ valid/ test/        # each with images/ and labels/ (YOLO polygon format, classes HR, PM, SPO2)
  ground_truth_vitals.csv    # columns: image, hr_true, spo2_true, confidence, legible, note
```

## Running

1. Open a pipeline notebook in Colab with a GPU runtime.
2. Run top to bottom. `EVAL_SPLIT` (default `test`) selects the split for evaluation.
3. Each notebook writes `pipeline_metrics_summary.csv` and
   `<pipeline>_coverage_bins.csv` into `DATASET_DIR`.
4. After all three have run, run `Pipeline_Comparison.ipynb` to produce Table 1
   and the coverage chart.

