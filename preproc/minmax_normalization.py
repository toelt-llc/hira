import numpy as np

def minmax_transform(arr):
    try:
        arr = np.asarray(arr)
    except ValueError:
        print("variable cannot be converted to numpy array")

    print(type(arr))
    assert type(arr) == np.ndarray

    new_list = [] # we store the images in a list and then convert it to an nparray for faster processing
    
    # check size of array, make sure we are taking one image at a time for minmax transformation
    print(arr.shape)
    if len(list(arr.shape)) >2:    
        
        for image in arr:
            im_std = (image - image.min()) / (image.max() - image.min())
            im_scaled = im_std * (1 - 0) + 0

            new_list.append(im_scaled)        

    else:
        arr_std = (arr - arr.min()) / (arr.max() - arr.min())
        arr_scaled = arr_std * (1 - 0) + 0

        new_list.append(arr_scaled)

    return np.asarray(new_list)
