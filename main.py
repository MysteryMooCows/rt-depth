from rich import print

import PIL
import depth_pro
import torch

import cv2
import matplotlib.pyplot as plt


def main():
    print("\ninvoked main()...\n")

    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda"),
                                                             precision=torch.float16)
    model.eval()

    # Load and preprocess an image.
    image, _, f_px = depth_pro.load_rgb("images\mario.webp")
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)

    print(f"Available tensors:\n{prediction.keys()}\n")

    depth = prediction["depth"].cpu().numpy()  # Depth in [m], copy to CPU.
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.

    print(f"depth: {depth}")
    print(f"focallength_px: {focallength_px}\n")

    plt.imshow(depth)
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()