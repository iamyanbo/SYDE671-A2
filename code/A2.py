# Yanbo Cheng
# 21287485
from pathlib import Path
import os
import time

from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d
import cv2
import imageio.v3 as iio
import numpy as np

def convolve_four_loops(input, kernel):
    # Zero padding
    # Have to account for non square kernels
    k_h = len(kernel)
    k_w = len(kernel[0])
    padding_len_h = (k_h - 1) // 2
    padding_len_w = (k_w - 1) // 2
    h = len(input) + 2 * padding_len_h
    w = len(input[0]) + 2 * padding_len_w
    padded_input = [[0] * w for _ in range(h)]
    for row in range(len(input)):
        for col in range(len(input[0])):
            padded_input[row + padding_len_h][col + padding_len_w] = input[row][col]

    # Convolution
    out_h = h - k_h + 1
    out_w = w - k_w + 1
    output = [[0] * out_w for _ in range(out_h)]
    for row in range(out_h):
        for col in range(out_w):
            total = 0
            for i in range(k_h):
                for j in range(k_w):
                    # Flip the kernel for convolution
                    total += padded_input[row + i][col + j] * kernel[k_h - 1 - i][k_w - 1 - j]
            output[row][col] = total
    return output

def convolve_two_loops(input, kernel):
    # Zero padding
    k_h = len(kernel)
    k_w = len(kernel[0])
    padding_len_h = (k_h - 1) // 2
    padding_len_w = (k_w - 1) // 2
    h = len(input) + 2 * padding_len_h
    w = len(input[0]) + 2 * padding_len_w
    padded_input = [[0] * w for _ in range(h)]
    for row in range(len(input)):
        for col in range(len(input[0])):
            padded_input[row + padding_len_h][col + padding_len_w] = input[row][col]

    # Convolution
    out_h = h - k_h + 1
    out_w = w - k_w + 1
    output = [[0] * out_w for _ in range(out_h)]
    # Flatten to 1D to reduce the number of loops
    for i in range(out_h * out_w):
        row = i // out_w
        col = i % out_w
        total = 0
        for j in range(k_h * k_w):
            kernel_row = j // k_w
            kernel_col = j % k_w
            total += padded_input[row + kernel_row][col + kernel_col] * kernel[k_h - 1 - kernel_row][k_w - 1 - kernel_col]
        output[row][col] = total
    return output

def gaussian_filter(image, kernel_size=5, sigma=-1, boundary='fill'):
    g = cv2.getGaussianKernel(kernel_size, sigma)
    # Apply the Gaussian vertically, then horizontally
    if image.ndim == 2:
        blurred = convolve2d(image, g, mode='same', boundary=boundary)
        return convolve2d(blurred, g.T, mode='same', boundary=boundary)

    # Filter each RGB channel separately
    copy_image = np.empty(image.shape, dtype=np.float64)
    for c in range(image.shape[2]):
        blurred = convolve2d(image[:, :, c], g, mode='same', boundary=boundary)
        copy_image[:, :, c] = convolve2d(blurred, g.T, mode='same', boundary=boundary)

    return copy_image

def unsharp_heuristic(image, kernel_size=5, alpha=1.5, sigma=-1, boundary='fill'):
    low_frequency = gaussian_filter(image, kernel_size, sigma, boundary=boundary)
    high_frequency = image - low_frequency
    sharpened_image = image + alpha * high_frequency
    return sharpened_image

def unsharp_single_convolution(image, kernel_size=5, alpha=1.5, sigma=-1, boundary='fill'):
    g = cv2.getGaussianKernel(kernel_size, sigma)
    G = g @ g.T
    identity_kernel = np.zeros((kernel_size, kernel_size))
    identity_kernel[kernel_size // 2, kernel_size // 2] = 1
    kernel = (1 + alpha) * identity_kernel - alpha * G
    if image.ndim == 2:
        return convolve2d(image, kernel, mode='same', boundary=boundary)

    sharpened_image = np.empty(image.shape, dtype=np.float64)
    for c in range(image.shape[2]):
        sharpened_image[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary=boundary)
    return sharpened_image


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    out_dir = project_dir / "filtered_output"
    os.makedirs(out_dir, exist_ok=True)

    # 1.1
    selfie = iio.imread(data_dir / "selfie.jpg", mode="L").astype(np.float64) / 255
    box_filter = np.ones((9, 9)) / 81
    Dx = np.array([[1, 0, -1]])
    Dy = Dx.T

    for name, kernel in [("Box", box_filter), ("Dx", Dx), ("Dy", Dy)]:
        outputs = []
        times = []
        for method in [convolve_four_loops, convolve_two_loops, convolve2d]:
            start = time.perf_counter()
            if method == convolve2d:
                output = method(selfie, kernel, mode="same", boundary="fill", fillvalue=0)
            else:
                output = np.array(method(selfie, kernel))
            times.append(time.perf_counter() - start)
            outputs.append(output)
        print(f"{name}: four loops {times[0]:.2f}s, two loops {times[1]:.2f}s, SciPy {times[2]:.2f}s")

        plt.figure(figsize=(12, 4))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(outputs[i], cmap="gray", vmin=0 if name == "Box" else -1, vmax=1)
            plt.title(["Four loops", "Two loops", "SciPy"][i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"convolution_{name}.png")
        plt.show()
        plt.close()
    iio.imwrite(out_dir / "selfie_gray.png", (selfie * 255).round().astype(np.uint8))

    # 1.2
    cameraman = iio.imread(data_dir / "cameraman.png", mode="L").astype(np.float64) / 255
    I_x = convolve2d(cameraman, Dx, mode="same", boundary="symm")
    I_y = convolve2d(cameraman, Dy, mode="same", boundary="symm")
    gradient_magnitude = np.sqrt(I_x ** 2 + I_y ** 2)
    threshold = 0.20  # Removes more grass and background texture
    edge_image = gradient_magnitude > threshold

    images = [cameraman, I_x, I_y]
    titles = ["Cameraman", "Horizontal", "Vertical"]
    plt.figure(figsize=(10, 4))
    for i in range(len(images)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap="gray", vmin=-1 if i in [1, 2] else 0, vmax=1)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "finite_differences.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    for i, image in enumerate([gradient_magnitude, edge_image]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image, cmap="gray", vmin=0, vmax=1)
        plt.title(["Magnitude", "Edges: threshold 0.20"][i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "finite_edges.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 4))
    for i, value in enumerate([0.05, 0.10, 0.20]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(gradient_magnitude > value, cmap="gray", vmin=0, vmax=1)
        plt.title(f"Threshold {value:.2f}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "edge_thresholds.png")
    plt.show()
    plt.close()

    # 1.3
    sigma = 2
    # Cover three standard deviations on each side
    kernel_size = 6 * sigma + 1
    g = cv2.getGaussianKernel(kernel_size, sigma)
    G = g @ g.T
    smoothed = gaussian_filter(cameraman, kernel_size, sigma, boundary="symm")
    smooth_x = convolve2d(smoothed, Dx, mode="same", boundary="symm")
    smooth_y = convolve2d(smoothed, Dy, mode="same", boundary="symm")
    smooth_magnitude = np.sqrt(smooth_x ** 2 + smooth_y ** 2)

    # DoG filters
    DoG_x = convolve2d(G, Dx, mode="full")
    DoG_y = convolve2d(G, Dy, mode="full")
    dog_x = convolve2d(cameraman, DoG_x, mode="same", boundary="symm")
    dog_y = convolve2d(cameraman, DoG_y, mode="same", boundary="symm")
    dog_magnitude = np.sqrt(dog_x ** 2 + dog_y ** 2)
    plt.figure(figsize=(10, 3))
    for i, kernel in enumerate([G, DoG_x, DoG_y]):
        plt.subplot(1, 3, i + 1)
        limit = np.max(np.abs(kernel))
        plt.imshow(kernel, cmap="gray", vmin=-limit, vmax=limit)
        plt.title(["Gaussian", "DoG x", "DoG y"][i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "dog_filters.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    for i, image in enumerate([cameraman, smoothed]):
        plt.subplot(1, 2, i + 1)
        plt.imshow(image, cmap="gray", vmin=0, vmax=1)
        plt.title(["Original", "Gaussian blur: sigma 2"][i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "gaussian_blur.png")
    plt.show()
    plt.close()

    # Horizontal, vertical, magnitude, and edges
    for name, horizontal, vertical, magnitude in [
        ("gaussian_derivatives", smooth_x, smooth_y, smooth_magnitude),
        ("dog_results", dog_x, dog_y, dog_magnitude)
    ]:
        images = [horizontal, vertical, magnitude, magnitude > 0.10]
        titles = ["Horizontal", "Vertical", "Magnitude", "Edges: threshold 0.10"]
        plt.figure(figsize=(8, 8))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(images[i], cmap="gray", vmin=-1 if i < 2 else 0, vmax=1)
            plt.title(titles[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png")
        plt.show()
        plt.close()

    plt.figure(figsize=(10, 4))
    images = [edge_image, smooth_magnitude > 0.10, dog_magnitude > 0.10]
    titles = ["Finite differences: 0.20", "Blur then differences: 0.10", "DoG: 0.10"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "dog_comparison.png")
    plt.show()
    plt.close()

    # Bells and whistles
    angle = np.arctan2(I_y, I_x)
    hue = (angle + np.pi) / (2 * np.pi)
    brightness = np.sqrt(gradient_magnitude / gradient_magnitude.max())
    hsv = np.stack([hue, np.ones_like(hue), brightness], axis=2)
    orientation = hsv_to_rgb(hsv)
    iio.imwrite(out_dir / "gradient_orientation.png", (orientation * 255).round().astype(np.uint8))
    plt.figure(figsize=(6, 6))
    plt.imshow(orientation)
    plt.title("Orientation: hue; strength: brightness")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "gradient_orientation_figure.png")
    plt.show()
    plt.close()

    # 2.1
    for filename, name in [("taj.jpg", "Taj_Mahal"), ("architecture.png", "Architecture")]:
        image = iio.imread(data_dir / filename, mode="RGB").astype(np.float64) / 255
        if image.shape[1] > 900:
            image = cv2.resize(image, (900, round(image.shape[0] * 900 / image.shape[1])))
        blurred = gaussian_filter(image, kernel_size=13, sigma=2, boundary="symm")
        high = image - blurred
        sharpened = unsharp_heuristic(image, 13, alpha=1, sigma=2, boundary="symm")
        stronger = unsharp_single_convolution(image, 13, alpha=2, sigma=2, boundary="symm")
        # Shift zero to gray so negative values are visible
        images = [image, blurred, high + 0.5, sharpened, stronger]
        titles = ["Original", "Blur: sigma 2", "High frequencies + 0.5", "Sharpened: alpha 1", "Sharpened: alpha 2"]
        plt.figure(figsize=(16, 5))
        for i in range(len(images)):
            plt.subplot(1, 5, i + 1)
            plt.imshow(np.clip(images[i], 0, 1))
            plt.title(titles[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_sharpening.png", dpi=160, bbox_inches="tight")
        plt.show()
        plt.close()

    # Evaluation
    sharp = iio.imread(data_dir / "architecture.png", mode="RGB").astype(np.float64) / 255
    blurred = gaussian_filter(sharp, 13, 2, boundary="symm")
    recovered = unsharp_single_convolution(blurred, 13, alpha=1, sigma=2, boundary="symm")
    images = [sharp, blurred, recovered]
    titles = ["Sharp input", "Deliberately blurred", "Resharpened: alpha 1"]
    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "sharpening_recovery.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    # 2.2
    derek = iio.imread(data_dir / "DerekPicture.jpg", mode="L").astype(np.float32) / 255
    nutmeg = iio.imread(data_dir / "nutmeg.jpg", mode="L").astype(np.float32) / 255

    # Manually selected corresponding eye positions (x, y)
    derek_pts = np.float32([[300, 343], [440, 332]])
    nutmeg_pts = np.float32([[604, 279], [741, 365]])

    # Align Nutmeg to Derek
    M, _ = cv2.estimateAffinePartial2D(nutmeg_pts, derek_pts)
    nutmeg_aligned = cv2.warpAffine(nutmeg, M, (derek.shape[1], derek.shape[0]))

    # Manually selected crop region
    derek = derek[145:680, 160:580]
    nutmeg_aligned = nutmeg_aligned[145:680, 160:580]

    sigma_low = 8
    low_frequency = gaussian_filter(derek, kernel_size=6 * sigma_low + 1, sigma=sigma_low, boundary="symm")
    sigma_high = 4
    nutmeg_blurred = gaussian_filter(nutmeg_aligned, kernel_size=6 * sigma_high + 1, sigma=sigma_high, boundary="symm")
    high_frequency = nutmeg_aligned - nutmeg_blurred
    hybrid = low_frequency + high_frequency

    images = [derek, low_frequency, nutmeg_aligned, high_frequency]
    titles = ["Derek", "Derek low frequencies", "Nutmeg", "Nutmeg high frequencies"]

    plt.figure(figsize=(8, 10))
    for i in range(len(images)):
        plt.subplot(2, 2, i + 1)
        display = images[i]
        if i == 3:
            display = display + 0.5
        plt.imshow(display, cmap="gray", vmin=0, vmax=1)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "Derek_Nutmeg_process.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    # Compare sigmas
    plt.figure(figsize=(10, 12))
    for row, low_sigma in enumerate([4, 8, 16]):
        low_candidate = gaussian_filter(derek, 6 * low_sigma + 1, low_sigma, boundary="symm")
        for col, high_sigma in enumerate([2, 4, 8]):
            high_candidate = nutmeg_aligned - gaussian_filter(nutmeg_aligned, 6 * high_sigma + 1, high_sigma, boundary="symm")
            plt.subplot(3, 3, row * 3 + col + 1)
            plt.imshow(np.clip(low_candidate + high_candidate, 0, 1), cmap="gray", vmin=0, vmax=1)
            plt.title(f"Low sigma {low_sigma}, high sigma {high_sigma}")
            plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "Derek_Nutmeg_cutoff_comparison.png")
    plt.show()
    plt.close()

    # Fourier transforms
    spectra = []
    images = [derek, low_frequency, nutmeg_aligned, high_frequency, hybrid]
    titles = ["Derek", "Derek low", "Nutmeg", "Nutmeg high", "Hybrid"]
    for image in images:
        fft = np.fft.fftshift(np.fft.fft2(image))
        spectra.append(np.log1p(np.abs(fft)))
    spectrum_max = max(spectrum.max() for spectrum in spectra)

    plt.figure(figsize=(15, 5))
    for i in range(len(spectra)):
        plt.subplot(1, 5, i + 1)
        plt.imshow(spectra[i], cmap="gray", vmin=0, vmax=spectrum_max)
        plt.title(titles[i] + " FFT")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "Derek_Nutmeg_fourier.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()

    iio.imwrite(out_dir / "hybrid.jpg", (np.clip(hybrid, 0, 1) * 255).round().astype(np.uint8))

    pairs = [
        ("obama.jpg", "jobs.jpg", "Obama_Jobs"),
        ("biden.jpg", "trump.jpg", "Biden_Trump")
    ]

    for low_path, high_path, name in pairs:
        low_img = iio.imread(data_dir / low_path, mode="RGB").astype(np.float32) / 255
        high_img = iio.imread(data_dir / high_path, mode="RGB").astype(np.float32) / 255

        if name == "Obama_Jobs":
            # Manually selected eyes and mouth centre
            low_pts = np.float32([[225, 209], [301, 206], [263, 297]])
            high_pts = np.float32([[160, 225], [273, 223], [218, 345]])
            x1, y1, x2, y2 = 150, 65, 365, 355
            sigma_low = 3
            sigma_high = 4
        else:  # Biden_Trump
            low_pts = np.float32([[211, 181], [270, 181]])
            high_pts = np.float32([[203, 213], [277, 212]])
            x1, y1, x2, y2 = 150, 50, 330, 305
            sigma_low = 4
            sigma_high = 4

        if name == "Obama_Jobs":
            M = cv2.getAffineTransform(high_pts, low_pts)
        else:
            M, _ = cv2.estimateAffinePartial2D(high_pts, low_pts)
        high_aligned = cv2.warpAffine(high_img, M, (low_img.shape[1], low_img.shape[0]))
        low_img = low_img[y1:y2, x1:x2]
        high_aligned = high_aligned[y1:y2, x1:x2]

        low_gray = cv2.cvtColor(low_img, cv2.COLOR_RGB2GRAY)
        high_gray = cv2.cvtColor(high_aligned, cv2.COLOR_RGB2GRAY)

        # Low and high frequencies
        kernel_low = 6 * sigma_low + 1
        kernel_high = 6 * sigma_high + 1
        low_frequency = gaussian_filter(low_gray, kernel_low, sigma_low, boundary="symm")
        high_blurred = gaussian_filter(high_gray, kernel_high, sigma_high, boundary="symm")
        high_frequency = high_gray - high_blurred
        hybrid = np.clip(low_frequency + high_frequency, 0, 1)

        images = [low_gray, high_gray, low_frequency, high_frequency + 0.5, hybrid]
        titles = ["Low Input", "High Input", "Low Frequency", "High Frequency", "Hybrid"]
        plt.figure(figsize=(15, 5))
        for i in range(len(images)):
            plt.subplot(1, 5, i + 1)
            plt.imshow(images[i], cmap="gray", vmin=0, vmax=1)
            plt.title(titles[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_process.png")
        plt.show()
        plt.close()

        iio.imwrite(out_dir / f"{name}_hybrid.png", (hybrid * 255).round().astype(np.uint8))

        plt.figure(figsize=(8, 5))
        for i, image in enumerate([low_frequency, high_frequency + 0.5]):
            plt.subplot(1, 2, i + 1)
            plt.imshow(image, cmap="gray", vmin=0, vmax=1)
            plt.title(["Low frequencies", "High frequencies + 0.5"][i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_components.png")
        plt.show()
        plt.close()

        # 2.2 Bells and Whistles
        low_color = gaussian_filter(low_img, kernel_low, sigma_low, boundary="symm")
        high_color = high_aligned - gaussian_filter(high_aligned, kernel_high, sigma_high, boundary="symm")
        low_gray_rgb = np.repeat(low_frequency[:, :, None], 3, axis=2)
        high_gray_rgb = np.repeat(high_frequency[:, :, None], 3, axis=2)

        color_images = [
            low_gray_rgb + high_gray_rgb,
            low_color + high_gray_rgb,
            low_gray_rgb + high_color,
            low_color + high_color
        ]
        titles = ["Both Grayscale", "Low-pass Color", "High-pass Color", "Both Color"]

        plt.figure(figsize=(12, 5))
        for i in range(len(color_images)):
            plt.subplot(1, 4, i + 1)
            plt.imshow(np.clip(color_images[i], 0, 1))
            plt.title(titles[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_color_comparison.png")
        plt.show()
        plt.close()

    # 2.3
    apple = iio.imread(data_dir / "burt_apple.png", mode="RGB").astype(np.float64) / 255
    orange = iio.imread(data_dir / "burt_orange.png", mode="RGB").astype(np.float64) / 255

    # Apple on the left and orange on the right
    mask = np.zeros_like(apple)
    mask[:, :apple.shape[1] // 2] = 1

    gaussian_stacks = []
    for image in [apple, orange, mask]:
        gaussian_stack = [image]
        for level in range(1, 6):
            sigma = 2 ** level
            kernel_size = 6 * sigma + 1
            blurred = gaussian_filter(image, kernel_size, sigma, boundary="symm")
            gaussian_stack.append(blurred)

        gaussian_stacks.append(gaussian_stack)

    laplacian_stacks = []
    for gaussian_stack in gaussian_stacks[:2]:
        laplacian_stack = []
        for level in range(len(gaussian_stack) - 1):
            laplacian_stack.append(gaussian_stack[level] - gaussian_stack[level + 1])
        # Keep the final blur so the levels sum back to the original
        laplacian_stack.append(gaussian_stack[-1])

        laplacian_stacks.append(laplacian_stack)

    gaussian_apple, gaussian_orange, gaussian_mask = gaussian_stacks
    laplacian_apple, laplacian_orange = laplacian_stacks

    for name, gaussian_stack, laplacian_stack in [
        ("Apple", gaussian_apple, laplacian_apple),
        ("Orange", gaussian_orange, laplacian_orange)
    ]:
        plt.figure(figsize=(18, 6))
        for level in range(len(gaussian_stack)):
            plt.subplot(2, 6, level + 1)
            plt.imshow(np.clip(gaussian_stack[level], 0, 1))
            plt.title(f"Gaussian {level}")
            plt.axis("off")

            plt.subplot(2, 6, level + 7)
            if level == len(laplacian_stack) - 1:
                plt.imshow(np.clip(laplacian_stack[level], 0, 1))
                plt.title("Residual")
            else:
                plt.imshow(np.clip(laplacian_stack[level] + 0.5, 0, 1))
                plt.title(f"Laplacian {level}")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_stacks.png")
        plt.show()
        plt.close()

    masked_apple = []
    masked_orange = []
    for level in range(len(laplacian_apple)):
        masked_apple.append(gaussian_mask[level] * laplacian_apple[level])
        masked_orange.append((1 - gaussian_mask[level]) * laplacian_orange[level])

    apple_contribution = np.sum(masked_apple, axis=0)
    orange_contribution = np.sum(masked_orange, axis=0)
    oraple = apple_contribution + orange_contribution

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]
    plt.figure(figsize=(10, 12))
    for row, level in enumerate([0, 2, 4]):
        bands = [masked_apple[level], masked_orange[level],
                 masked_apple[level] + masked_orange[level]]
        for col in range(3):
            plt.subplot(4, 3, row * 3 + col + 1)
            plt.imshow(np.clip(bands[col] + 0.5, 0, 1))
            plt.title(f"{labels[row * 3 + col]} Level {level}")
            plt.axis("off")

    for col, image in enumerate([apple_contribution, orange_contribution, oraple]):
        plt.subplot(4, 3, 10 + col)
        plt.imshow(np.clip(image, 0, 1))
        plt.title(["(j) Apple contribution", "(k) Orange contribution", "(l) Oraple"][col])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "Figure_3_42.png")
    plt.show()
    plt.close()

    iio.imwrite(out_dir / "oraple.png", (np.clip(oraple, 0, 1) * 255).round().astype(np.uint8))

    # 2.4
    pairs = [("obama.jpg", "jobs.jpg", "Obama_Jobs"),
             ("biden.jpg", "trump.jpg", "Biden_Trump")]

    for background_path, face_path, name in pairs:
        background = iio.imread(data_dir / background_path, mode="RGB").astype(np.float64) / 255
        face = iio.imread(data_dir / face_path, mode="RGB").astype(np.float64) / 255

        # Reuse the points and crops from 2.2
        if name == "Obama_Jobs":
            background_pts = np.float32([[225, 209], [301, 206], [263, 297]])
            face_pts = np.float32([[160, 225], [273, 223], [218, 345]])
            x1, y1, x2, y2 = 150, 65, 365, 355
            centre = (110, 165)
            radius = 85
        else:
            background_pts = np.float32([[211, 181], [270, 181]])
            face_pts = np.float32([[203, 213], [277, 212]])
            x1, y1, x2, y2 = 150, 50, 330, 305
            centre = (85, 160)
            radius = 60

        if name == "Obama_Jobs":
            M = cv2.getAffineTransform(face_pts, background_pts)
        else:
            M, _ = cv2.estimateAffinePartial2D(face_pts, background_pts)
        face = cv2.warpAffine(face, M, (background.shape[1], background.shape[0]))
        background = background[y1:y2, x1:x2]
        face = face[y1:y2, x1:x2]

        # Face mask
        mask = np.zeros(background.shape[:2], dtype=np.float64)
        cv2.circle(mask, centre, radius, 1, -1)
        mask = np.repeat(mask[:, :, None], 3, axis=2)
        soft_mask = gaussian_filter(mask, 13, 2, boundary="symm")

        # Same Gaussian and Laplacian stacks as 2.3
        gaussian_stacks = []
        for image in [face, background, soft_mask]:
            gaussian_stack = [image]
            for level in range(1, 6):
                sigma = 2 ** level
                kernel_size = 6 * sigma + 1
                gaussian_stack.append(gaussian_filter(image, kernel_size, sigma, boundary="symm"))
            gaussian_stacks.append(gaussian_stack)

        laplacian_stacks = []
        for gaussian_stack in gaussian_stacks[:2]:
            laplacian_stack = []
            for level in range(len(gaussian_stack) - 1):
                laplacian_stack.append(gaussian_stack[level] - gaussian_stack[level + 1])
            laplacian_stack.append(gaussian_stack[-1])
            laplacian_stacks.append(laplacian_stack)

        laplacian_face, laplacian_background = laplacian_stacks
        gaussian_mask = gaussian_stacks[2]
        masked_face = []
        masked_background = []
        for level in range(len(laplacian_face)):
            masked_face.append(gaussian_mask[level] * laplacian_face[level])
            masked_background.append((1 - gaussian_mask[level]) * laplacian_background[level])

        face_contribution = np.sum(masked_face, axis=0)
        background_contribution = np.sum(masked_background, axis=0)
        blended = face_contribution + background_contribution
        direct_cut = mask * face + (1 - mask) * background

        images = [background, face, mask, direct_cut, blended]
        titles = ["Background", "Aligned face", "Mask", "Direct cut", "Stack blend"]
        plt.figure(figsize=(15, 5))
        for i in range(len(images)):
            plt.subplot(1, 5, i + 1)
            plt.imshow(np.clip(images[i], 0, 1))
            plt.title(titles[i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_Circle_blend_comparison.png")
        plt.show()
        plt.close()
        iio.imwrite(out_dir / f"{name}_Circle_blend.png", (np.clip(blended, 0, 1) * 255).round().astype(np.uint8))

        plt.figure(figsize=(8, 5))
        for i, image in enumerate([mask * face, (1 - mask) * background]):
            plt.subplot(1, 2, i + 1)
            plt.imshow(image)
            plt.title(["Inserted face inside mask", "Background outside mask"][i])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_masked_inputs.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(10, 12))
        for row, level in enumerate([0, 2, 4]):
            bands = [masked_face[level], masked_background[level],
                     masked_face[level] + masked_background[level]]
            for col in range(3):
                plt.subplot(4, 3, row * 3 + col + 1)
                plt.imshow(np.clip(bands[col] + 0.5, 0, 1))
                plt.title(f"{['Face', 'Background', 'Combined'][col]} level {level}")
                plt.axis("off")
        for col, image in enumerate([face_contribution, background_contribution, blended]):
            plt.subplot(4, 3, 10 + col)
            plt.imshow(np.clip(image, 0, 1))
            plt.title(["Face contribution", "Background contribution", "Final blend"][col])
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_blend_process.png")
        plt.show()
        plt.close()
