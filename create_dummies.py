from PIL import Image
import os

def create_dummy_image(path, color):
    img = Image.new('RGB', (224, 224), color=color)
    img.save(path)
    print(f"Created {path}")

create_dummy_image('dummy_id.png', 'red')
create_dummy_image('dummy_face.png', 'blue')
