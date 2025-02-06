import os
import imageio.v2 as iio
from PIL import Image

def resize_images(folder_path, size):
    for file in os.listdir(folder_path):
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(folder_path, file)
            img = Image.open(image_path)
            img = img.resize(size, Image.LANCZOS)
            img.save(image_path)

def create_gif_from_images(folder_path, output_path, duration=0.1):  # Using `duration` correctly
    # Resize images to the same size
    first_image_path = os.path.join(folder_path, os.listdir(folder_path)[0])
    first_image = Image.open(first_image_path)
    size = first_image.size
    resize_images(folder_path, size)

    # List all files in the folder
    files = sorted(os.listdir(folder_path))
    images = []

    for file in files:
        if file.endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(folder_path, file)
            images.append(iio.imread(image_path))

    # Save images as a gif with a specific frame duration
    iio.mimsave(output_path, images, duration=duration)  # Adjust `duration` to control speed


def make_gif_from_one_folder():
    folder_path = '../../../../storage_nobackup/rnd_agents_storage/Scn__2024_07_20_1414__FTC__XRr__HRes__ZSLR__RND_Off_Light'
    output_path = f'{folder_path}/video_july2024_Onlight.gif'
    create_gif_from_images(folder_path, output_path, duration=0.1) 



#%%
def combine_images(img1, img2):
    combined = Image.new('RGB', (img1.width + img2.width, img1.height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    return combined

def create_gif_from_images_of_two_folders(folder_path1, folder_path2, output_path, duration=0.1):
    # Resize images to the same size
    first_image_path1 = os.path.join(folder_path1, os.listdir(folder_path1)[0])
    first_image1 = Image.open(first_image_path1)
    size1 = first_image1.size
    resize_images(folder_path1, size1)
    
    first_image_path2 = os.path.join(folder_path2, os.listdir(folder_path2)[0])
    first_image2 = Image.open(first_image_path2)
    size2 = first_image2.size
    resize_images(folder_path2, size2)
    
    # Ensure both sizes are the same
    if size1 != size2:
        raise ValueError("Images in both folders must be of the same size after resizing")

    # List and sort all files in both folders
    files1 = sorted([f for f in os.listdir(folder_path1) if f.endswith(('png', 'jpg', 'jpeg'))])
    files2 = sorted([f for f in os.listdir(folder_path2) if f.endswith(('png', 'jpg', 'jpeg'))])

    # Ensure both folders have the same number of images
    if len(files1) != len(files2):
        raise ValueError("Both folders must contain the same number of images")

    images = []

    for file1, file2 in zip(files1, files2):
        image_path1 = os.path.join(folder_path1, file1)
        image_path2 = os.path.join(folder_path2, file2)
        
        img1 = Image.open(image_path1)
        img2 = Image.open(image_path2)
        
        combined_image = combine_images(img1, img2)
        
        images.append(combined_image)


    if out_format == 'gif':
        # Save images as a gif with a specific frame duration
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration*1000, loop=0)
    elif out_format == 'mpf':
        writer = iio.get_writer(output_path, fps=fps)
    for img in images:
        writer.append_data(iio.imread(img))
    writer.close()



def make_gif_from_one_folder():
    
    create_gif_from_images_of_two_folders(folder_path1, folder_path2, output_path, duration=duration)
    
    
    
if __name__ == '__name__':
    folder1 = 'Scn__2024_07_20_1414__FTC__XRr__HRes__ZSLR__RND_Off_Light'
    folder2 = 'Scn__2024_07_20_1414__FTC__XRr__HRes__ZSLR__RND_On_Light'
    folder_path1 = f'../../../../storage_nobackup/rnd_agents_storage/{folder1}'
    folder_path2 = f'../../../../storage_nobackup/rnd_agents_storage/{folder2}'
    folders = [folder_path1, folder_path2]
    duration = 2 # second
    output_path = f'{folder_path1}/video_july2024_Combined_{duration}s.gif'
    
    if out_format == 'gif':
        # Save images as a gif with a specific frame duration
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration*1000, loop=0)
         
    elif out_format == 'mp4':
        writer = iio.get_writer(output_path, fps=fps)
        for img in images:
            writer.append_data(iio.imread(img))
        writer.close()

