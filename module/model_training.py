from ultralytics import YOLO

# Charger un modèle pré-entraîné pour detection
model = YOLO('yolov8n.pt')  # 'n' = nano model, rapide pour débuter

# Entraîner sur ton dataset
model.train(
    data='silkworm.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='silkworm-detector'
)
