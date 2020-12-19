# CSE-490-G1

# Example Project
In this project, I tackled semantic image segmentation. I used a U-net architecture and trained on the 2007 Pascal VOC image segmentation dataset. I got up to 75% average pixel accuracy, but this is highly misleading, as I'll discuss in more detail below.

# Introduction
Semantic image segmentation involves breaking images down into their basic parts. The goal is to match each pixel with its correspondig group. In this way, a certain level of sophistication is required. The AI must not only be able to recognize the class or type of an object in the picture, but also its shape and parts. One important application of image segmentation is driverless cars. In order to move about on the road, cars need to know what objects are around. Running over a branch is very different from running over a kitten. Another application is medical imaging. AI could potentially spot diseases before they become serious and assist doctors.

# Related Work
I reused a lot of code from the CifarNet in homework 1. I did not use any code from U-nets or image segmentation nets from online. One of the early ideas in image segmentation was the encoder-decoder scheme, building upon the advancements in classification. Another was the U-net, which I have in this project, and another more recent development is spatial pyramid pooling, which allows the classifier to look across different layers of the image to find different objects.

# Approach
The Pascal VOC dataset has 20 classes. I ended up approaching this problem as a pixel-level classification problem. The neural network predicts an image of the same size as the input. But instead of having 3 channels, it has 21, where the extra channel is for predicting backgrounds. In this layout, the network predicts what can be thought of as a distribution across the channels for each pixel. The channel in which it places most of the weight is taken to be the predicted label for the pixel. I could use cross-entropy as a loss function, passing in values for all the pixels rather than just one for the image.

For architecture, I choses to use the U-net, which was discussed in class. The U-net has an encoder-decoder format, enabling it to carry meaningful semantical information while still being big enough to predict granular details. The U-net also has the advantage of having skip connections, which enable data to travel from the back of the net to the front. This is important, as having rich, semantical information often comes at the tradeoff of detailed, pixular information. These skip connections are a workaround, and also help speed up training by providing more ways for the error to backpropagate.

My net has three layers of strided convolutions, used to extract features from the image. These layers are then followed by three layers of deconvolutions. An advantage of deconvolutions compared to upsampling is that they learn weights and are not just simple repetitions. This helps the decoder to better interpret the state of the encoder. I used Relu activation functions to introduce non-linearity. For skip connections, I chose to add previous layers to the corresponding new ones, rather than packing them on. This seemed simplest, and it also helps prevent having super dense nets than can be difficult to train.

One difficulty I encountered in this project was that the images are of variable size. This added a challenge. For one, it makes linear fully-connected layers difficult to do, if not impossible, as the dimensions of the image are not all the same. For this same reason, I also had to train with batch sizes of 1. For the forward pass, I had to keep track of the original size of the image and manually adjust it back to be the same. Sometimes I added zero-padding if the result was too small, and sometimes I sliced off extra parts from the array. The issue is that the convolutions can map different input spaces to the same output. So I could not count on everything easily being undone on their own.

# Results
I evaluated my approach using pixel accuracy. My net got up to 75% accuracy. This may seem good, but is actually quite misleading. Especially in datasets such as Pascal VOC, where there are only a couple objects per image, there is a lot of background pixels. As a result, the classifier can get a good deal of pixels right simply by labeling things as background. My classifier did exactly this. Within a few epochs of training, the classifier learned to classify the entire image as background. This increased the accuracy quickly at first, but it then hit a wall where it would no longer improve.

# Discussion
I was not expecting how difficult it would be to train the net. I thought that once I got everything setup and the architecture done fairly well that the results would come, which they certainly did not. After training 80 epochs I was not able to make any more progress. I was also not expecting the classifier to predict everything as background. I know my evaluation metric was subpar, but this is unrelated. The training depends on the loss function, of which I used a multi-instance cross-entropy function. In the future, I think I need to use a larger net and train for longer periods of time. My net is small, especially compared to those Google and Facebook would use. Getting a pre-trained CNN on ImageNet would also definitely be something to consider, as training the encoder certainly takes a great deal of processing.
