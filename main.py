from rich import print

import PIL
from PIL import Image
import depth_pro
import torch

import cv2
import matplotlib.pyplot as plt


DEBUG = False


def dprint(msg):
    if DEBUG:
        print(msg)


def main():
    print("Initializing...")

    dprint("\ninvoked main()...\n")
    dprint(f"CUDA available: {torch.cuda.is_available()}")

    model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"),
                                                             precision=torch.float16)
    model.eval()

    dprint("Getting capture device...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[red][bold]Error: Could not open capture device[/bold][/red]")
        exit(1)

    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[red][bold]Error: Can't receive frame (stream end?). Exiting...[/bold][/red]")
                break

            image = transform(frame)

            prediction = model.infer(image) #, f_px=f_px)

            dprint(f"Available tensors:\n{prediction.keys()}\n")

            depth = prediction["depth"].cpu().numpy() * 0.25  # Depth in [m], copy to CPU.
            focallength_px = prediction["focallength_px"]  # Focal length in pixels.

            dprint(f"depth: {depth}")
            dprint(f"focallength_px: {focallength_px}\n")
            
            cv2.imshow('Depth', depth)
            
            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()