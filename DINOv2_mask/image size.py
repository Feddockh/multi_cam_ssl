from PIL import Image

img = Image.open(r'D:\vlr_projects\multi_cam_ssl\image_data\firefly_left\images\1739373917_960083712.png')
width, height = img.size
print(f"宽: {width}, 高: {height}")
