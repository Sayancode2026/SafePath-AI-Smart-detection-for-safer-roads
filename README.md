
# SafePath-AI-Smart-detection-for-safer-roads

This project uses YOLO (You Only Look Once) object detection models to detect 6 types of road surface damage. The model is trained using a dataset from **Roboflow** and tested on custom images and videos.

---

## 📁 Dataset Information

- **Source**: Roboflow  
- **License**: CC BY 4.0  
- **Roboflow URL**:  
  [https://universe.roboflow.com/baka-1ravj/road-damage-det/dataset/4](https://universe.roboflow.com/baka-1ravj/road-damage-det/dataset/4)

### 🔸 `data.yaml`
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['alligator cracking', 'edge cracking', 'longitudinal cracking', 'patching', 'rutting', 'transverse cracking']

roboflow:
  workspace: baka-1ravj
  project: road-damage-det
  version: 4
  license: CC BY 4.0
  url: https://universe.roboflow.com/baka-1ravj/road-damage-det/dataset/4
```

---

## 🧪 Testing the Model

```python
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

results = model.predict(source=image_path, conf=0.25, save=True)
```

Or test on a folder or video:

```python
model.predict(source="/content/test/images", conf=0.25, save=True)
```

---

## 📊 Evaluation Example

```
Box(P): 0.708
Recall: 0.659
mAP@0.5: 0.697
mAP@0.5:0.95: 0.464
```
