import matplotlib.pyplot as plt
import numpy as np

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray')
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

# helper function for data visualization 
def plot_sample(sample):
    """Plot SAR channel and flood mask"""
    image = sample['image']
    mask = sample['mask']
    image = image.transpose(1,2,0)
    visualize(vv=denormalize(image[:,:,0]),
                vh=denormalize(image[:,:,1]),
              mask=denormalize(mask))
    
# helper function for data visualization 
def plot_random(data_set):
    """Plot random sample"""
    n=np.random.randint(len(data_set))
    print(n)
    plot_sample(data_set[n])