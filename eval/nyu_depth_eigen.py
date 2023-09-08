import numpy as np
from torchvision import transforms

# Replace 'path_to_your_file.npz' with the actual file path
file_path = r'D:\eigen_split\nyu_test_001.npz'
to_PIL = transforms.ToPILImage()



    
data = np.load(file_path)


for array_name in data.files:
        array_data = data[array_name]
        print(f"{array_name}:")
        if array_name == 'image':
            image_data = array_data
            print("Image shape:", image_data.shape)
            to_PIL(image_data).show()
            
        elif array_name == 'depth':
            depth_data = array_data
            print("Depth shape:", depth_data.shape)
           
 


