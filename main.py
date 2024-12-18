import os
from rich import print as rprint

import PIL
from PIL import Image
import depth_pro
import torch
import numpy as np

import cv2
import matplotlib.pyplot as plt

from utils.console_io import ProgressIndicator

import matplotlib.pyplot as plt


DEBUG = False
RT = True
RT_SCALE = 1/4

IMG_PATH = os.path.join("images", "cat.webp")


def dprint(msg):
    if DEBUG:
        rprint(msg)

def main():
    dprint("\ninvoked main()...\n")
    dprint(f"CUDA available: {torch.cuda.is_available()}")

    rprint("Initializing", end="")

    with ProgressIndicator() as _:
        model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"),
                                                             precision=torch.float16)
        model.eval()

    rprint("[green]done[/green]")

    if RT:
        dprint("Getting capture device...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            rprint("[red][bold]Error: Could not open capture device[/bold][/red]")
            exit(1)

        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    rprint("[red][bold]Error: Can't receive frame (stream end?). Exiting...[/bold][/red]")
                    break

                image = transform(frame)

                prediction = model.infer(image) #, f_px=f_px)

                dprint(f"Available tensors:\n{prediction.keys()}\n")

                depth = prediction["depth"].cpu().numpy() * RT_SCALE  # Depth in [m]
                focallength_px = prediction["focallength_px"]

                dprint(f"depth: {depth}")
                dprint(f"focallength_px: {focallength_px}\n")
                
                cv2.imshow('Depth', depth)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    else:
        image, _, f_px = depth_pro.load_rgb(IMG_PATH)
        image = transform(image)

        prediction = model.infer(image, f_px=f_px)

        dprint(f"Available tensors:\n{prediction.keys()}\n")

        depth = prediction["depth"].cpu().numpy()  # Depth in [m]

        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

        focallength_px = prediction["focallength_px"]

        dprint(f"depth: {depth}")
        dprint(f"focallength_px: {focallength_px}\n")

        plt.imshow(depth)
        plt.colorbar()

        if not os.path.exists("output"):
            os.makedirs("output")

        plt.imsave(os.path.join("output", "depth.png"), depth)

        plt.show()


if __name__ == "__main__":
    main()