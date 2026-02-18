import numpy as np
import PIL.Image
import tensorflow as tf

#Loads pre-trained InceptionV3 model
model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")

#Chooses layer in model to enhance, gets tensor output of that layer, and outputs only that layer 
layer_name = 'mixed4d_3x3_bottleneck_pre_relu'
layer_output = model.get_layer(layer_name).output
dream_model = tf.keras.Model(inputs=model.input, outputs=layer_output)

#Converts image to array of floats so that math can be done with it
img0 = PIL.Image.open('pilatus800.jpg')
img0 = np.float32(img0)

#Defines loss function for DeepDream
def calc_loss(img):
    img_batch = tf.expand_dims(img, 0)
    activation = dream_model(img_batch)
    return tf.reduce_mean(activation)

#Function for applying DeepDream
def render_deepdream(img, steps=10, step_size=1.5):
    img = tf.convert_to_tensor(img)
    #Enhances the image
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = calc_loss(img)
        #Computes and normalizes gradient of the loss
        grads = tape.gradient(loss, img)
        grads /= tf.math.reduce_std(grads) + 1e-8
        #Moves image in direction of gradient
        img += grads * step_size
    #Returns dreamed image
    return img

#Calls deepdream function on original image
dream_img = render_deepdream(img0)
#Converts float image back to 8-bit integers for saving
result = np.uint8(np.clip(dream_img, 0, 255))
#Converts array back to image and saves it as png file
PIL.Image.fromarray(result).save('dream.png')

print("DeepDream complete!")
