"""
Apply a short-time Fourier transform to the data
Turn the transformed data into RGB images (similar to spectrograms)
Save the spectrograms (this will be disabled in the code, as the spectrograms have already been made)
Load the spectrograms to be ready for the convolutional network
"""

def complex2rgb(complex_array):
    """
    input is an array of complex numbers (stft conversion creates a complex array)
    output is an rgb array with depth = 3
    see the following links for more info on how these equations were used:
    https://en.wikipedia.org/wiki/Color_wheel_graphs_of_complex_functions
    https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSL
    """
    z = complex_array
    h = np.angle(z) / (2 * math.pi)
    l = 2 ** -(np.absolute(z))
    s = 1
    c = (1 - np.absolute(2 * l - 1))
    h_prime = h * 6.
    x = c * (1 - np.absolute(h_prime % 2 - 1))

    array_out = np.zeros_like(complex_array)

    mask1 = (h_prime >= 0) & (h_prime < 1) #(c, x, 0)
    mask2 = (h_prime >= 1) & (h_prime < 2) #(x, c, 0)
    mask3 = (h_prime >= 2) & (h_prime < 3) #(0, c, x)
    mask4 = (h_prime >= 3) & (h_prime < 4) #(0, x, c)
    mask5 = (h_prime >= 4) & (h_prime < 5) #(x, 0, c)
    mask6 = (h_prime >= 5) & (h_prime < 6) #(c, 0, x)

    r = (c * (mask1 | mask6)) + (x * (mask2 | mask5))
    g = (c * (mask2 | mask3)) + (x * (mask1 | mask4))
    b = (c * (mask4 | mask5)) + (x * (mask3 | mask6))

    return np.dstack((r, g, b))
    
def save_spectrograph():
    for i in range(len(X_train)):
        train_in = X_train[i]
        test_transform = librosa.core.stft(train_in)
        new_image = complex2rgb(test_transform)
        mpimg.imsave("./spectrograms/train_image_{0:04d}.png".format(i), new_image)

    for i in range(len(X_val)):
        val_in = X_val[i]
        val_transform = librosa.core.stft(val_in)
        new_image = complex2rgb(val_transform)
        mpimg.imsave("./spectrograms/val_image_{0:04d}.png".format(i), new_image)


    for i in range(len(X_test)):
        test_in = X_test[i]
        test_transform = librosa.core.stft(test_in)
        new_image = complex2rgb(test_transform)
        mpimg.imsave("./spectrograms/test_image_{0:04d}.png".format(i), new_image)
        
#save_spectrograph() #leave this disabled because the spectrograms have already been saved.

def load_spectrograph():
    train_out = []
    val_out = []
    test_out = []
    for i in range(len(X_train)):
        temp_train = mpimg.imread('./spectrograms/train_image_{0:04d}.png'.format(i))
        train_out.append(temp_train)
    train_out = np.array(train_out)
    
    for i in range(len(X_val)):
        temp_val = mpimg.imread('./spectrograms/val_image_{0:04d}.png'.format(i))
        val_out.append(temp_val)
    val_out = np.array(val_out)
    
    for i in range(len(X_test)):
        temp_test = mpimg.imread('./spectrograms/test_image_{0:04d}.png'.format(i))
        test_out.append(temp_test)
    test_out = np.array(test_out)
    
    print(train_out.shape)
    print(val_out.shape)
    print(test_out.shape)
    return train_out, val_out, test_out

train_out, val_out, test_out = load_spectrograph()
print("Finished Loading STFT Images.")
