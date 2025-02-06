import os
import imageio.v2 as iio
from PIL import Image, UnidentifiedImageError
import numpy as np  # Add this import

def resize_images(folder_path, size):
    """Resize all images in a folder to the specified size."""
    for file in os.listdir(folder_path):
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(folder_path, file)
            try:
                img = Image.open(image_path)
                img = img.resize(size, Image.LANCZOS)
                img.save(image_path)
            except UnidentifiedImageError:
                print(f"Cannot identify image file {image_path}, skipping.")

def combine_images(img1, img2):
    """Combine two images side by side."""
    combined = Image.new('RGB', (img1.width + img2.width, img1.height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    return combined

def gather_images(folder_paths):
    """Gather and resize images from one or two folders."""
    all_images = []
    sizes = []

    for folder_path in folder_paths:
        first_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        try:
            first_image = Image.open(first_image_path)
        except UnidentifiedImageError:
            print(f"Cannot identify first image file {first_image_path}, skipping folder.")
            continue
        
        size = first_image.size
        sizes.append(size)
        resize_images(folder_path, size)
        images = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('png', 'jpg', 'jpeg'))])
        all_images.append(images)
    
    if len(sizes) == 2 and sizes[0] != sizes[1]:
        raise ValueError("Images in both folders must be of the same size after resizing")

    if len(all_images) == 2 and len(all_images[0]) != len(all_images[1]):
        raise ValueError("Both folders must contain the same number of images")

    return all_images

def create_combined_images(images1, images2):
    """Create combined images from two sets of images."""
    combined_images = []
    for img1_path, img2_path in zip(images1, images2):
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            combined_image = combine_images(img1, img2)
            combined_images.append(combined_image)
        except UnidentifiedImageError as e:
            print(f"Cannot identify image file: {e}, skipping.")
    return combined_images

def save_as_gif(images, output_path, duration):
    """Save a list of images as a GIF."""
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration*1000, loop=0)

def save_as_mp4(images, output_path, fps):
    """Save a list of images as an MP4 video."""
    writer = iio.get_writer(output_path, fps=fps)
    for img in images:
        writer.append_data(np.array(img))  # Convert PIL Image to numpy array
    writer.close()

def create_media_from_images(folder_paths, output_path, duration, out_format='gif', fps=10):
    """Create a media file (GIF or MP4) from images in one or two folders."""
    all_images = gather_images(folder_paths)
    if len(all_images) == 2:
        images = create_combined_images(all_images[0], all_images[1])
    else:
        images = [Image.open(img) for img in all_images[0]]

    if out_format == 'gif':
        save_as_gif(images, output_path, duration)
    elif out_format == 'mp4':
        save_as_mp4(images, output_path, fps)

def main():
    folder1 = 'Scn__2024_07_20_1414__FTC__XRr__HRes__ZSLR__RND_Off_Light'
    folder2 = 'Scn__2024_07_20_1414__FTC__XRr__HRes__ZSLR__RND_On_Light'
    rnd_agents_storage = '../../../../storage_nobackup/rnd_agents_storage'
    folder_path1 = f'{rnd_agents_storage}/{folder1}'
    folder_path2 = f'{rnd_agents_storage}/{folder2}'
    folders = [folder_path1, folder_path2]
    
    # Configuration
    duration = 2  # seconds
    out_format = 'mp4'  # 'gif' or 'mp4'
    fps = 1  # Frames per second for mp4
    name_postfix = f'{duration}s' if out_format == 'gif' else f'{fps}fps'
    output_path = f'{rnd_agents_storage}/video_july2024_{name_postfix}.{out_format}'
    
    create_media_from_images(folders, output_path, duration, out_format, fps)

if __name__ == '__main__':
    main()
