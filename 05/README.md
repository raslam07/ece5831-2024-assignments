YouTube Link: https://youtu.be/WmNQNH3lPjM

In the 'teachable.ipynb' file, I first imported all the necessary packages and made sure they all worked properly by changing the environment.
Then, I loaded the trained model which was the .h5 file I obtained from the teachable machine website, along with the labels which were in a .txt file.
Next, I trained the model using one specific image. In my case, I used the image 16.jpg from the paper samples. After running the trained model, the output showed that the class was paper, and the confidence score was about 0.99.
Finally, I was able to output the actual image used in the testing by using matplotlib.

In the 'rock-paper-scissors.py' file, I organized the practice I did in the .ipynb file to different functions. For example, I had separate functions for loading the image, loading the model, preparing the input, making a prediction, and displaying the image.
When running the python file from the command line, I passed in the following argument: 'samples/paper/16.jpg'. This command line argument would be 'sys.argv[1]' as was specified in the code.

After running the 'rock-paper-scissors-live.py' file, I was able to access the camera on my laptop so that the model could make real time predictions about whether my hand was either a rock, paper, or scissor. I also ran this file from the command line, and the results of this can be shown in the YouTube video I have provided the link for.
