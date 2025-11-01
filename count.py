from ultralytics import YOLO
import sys
import cv2

MODEL = "runs/detect/silkworm-detector4/weights/best.pt"

def count_silkworms(image_path, model_path=MODEL, show_image=False):
    # Charger le modèle YOLO entraîné
    model = YOLO(model_path)

    # Inférer sur l'image
    results = model(image_path)

    # Extraire les boîtes détectées
    num_silkworms = len(results[0].boxes)

    print(f"🪱 Nombre estimé de vers à soie détectés : {num_silkworms}")

    if show_image:
        # Afficher l'image annotée
        annotated_image = results[0].plot()
        cv2.imshow("Vers à soie détectés", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return num_silkworms


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️  Usage : python count_silkworms.py <image_path> [model_path]")
        print("Exemple : python count_silkworms.py test.jpg best.pt")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else MODEL

    count_silkworms(image_path, model_path, show_image=True)
