# AI-Based Brain Tumor Detection Using MRI Images
## Dataset
Kaggle: Brain Tumor (MRI) Detection by Arwa Basal
https://www.kaggle.com/datasets/arwabasal/brain-tumor-mri-detection
The dataset page describes `yes` and `no` classes and an MIT license. Verify the current dataset card/license before submission.

## Model
MobileNetV2 transfer learning, 224x224 images, augmentation, frozen-backbone training, limited fine-tuning, sigmoid binary classifier.

## Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
python -m streamlit run app.py
```
The training script downloads the public dataset with `kagglehub`. Do not invent metrics: use `models/metrics.json` after training.

## Evaluation
Report accuracy, precision, recall, F1-score and confusion matrix.

## Medical disclaimer
This is an academic prototype and must not be used for diagnosis or treatment decisions.
