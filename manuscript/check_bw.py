import os
from PIL import Image

def convert_to_bw(image_path, output_path):
    print(f"Checking B&W readability for {image_path}...")
    try:
        with Image.open(image_path) as img:
            bw_img = img.convert('L') # Convert to grayscale
            bw_img.save(output_path)
            print(f"Successfully converted to B&W: {output_path}")
            print("Please visually inspect the B&W image to ensure all lines and legends are distinguishable.")
    except Exception as e:
        print(f"Error converting {image_path}: {e}")

if __name__ == "__main__":
    img_path = "cumulative_net_returns.png"
    out_path = "cumulative_net_returns_bw.png"
    if os.path.exists(img_path):
        convert_to_bw(img_path, out_path)
    else:
        print(f"Image not found: {img_path}")
