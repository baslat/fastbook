from fastai.vision.all import *
from fastbook import *

# Get the data
# it is path/train/number
path = untar_data(URLs.MNIST)
path.ls()

# ok, we will try a big version of pixel similarity
# and I will eat some humble pie and write filthy code

# so I need to get the mean image of each number
# get the list of subfolders, one for each digit
path_train = (path/'training').ls().sorted()
path_test = (path/'testing').ls().sorted()


# for each subfolder, read in the images into a tensor
# first define a function to list contents in a subfolder


def folder_to_tensor(folder):
    paths = folder.ls().sorted()
    return [tensor(Image.open(img)) for img in paths]

# Read in each subfolder of images


trainers = [folder_to_tensor(subfolder) for subfolder in path_train]
testers = [folder_to_tensor(subfolder) for subfolder in path_test]

# yay that worked!, so now I have a list where each element is a tensor of images

# For each element in the list, get the mean image

# We need to case the integers to floats too
stacks_train = [torch.stack(digit).float()/255 for digit in trainers]
stacks_test = [torch.stack(digit).float()/255 for digit in testers]
# Now turn into means (oh what I would do with pipes here)
means = [stack.mean(0) for stack in stacks_train]


show_image(means[5])

# Ok, now I have the means of each image. I need to calculate a loss function
# I will use RMSE, which is already available as a function

# Get a number to test
a_5 = trainers[5][1]
a_7 = trainers[7][100]

F.mse_loss(a_7, means[5]).sqrt()
F.mse_loss(a_7, means[8]).sqrt()
F.mse_loss(a_7, means[7]).sqrt()

# all very close.
# lets write a function that calcs the distance against all means for a single number, and returns the index of the closest
x = a_5


def closest(x):
    dists = [F.mse_loss(x, avgs).sqrt() for avgs in means]
    return dists.index(min(dists))


def guessed_right(guess, truth):
    return guess == truth


a_8 = stacks_test[8][5]

guessed_right(closest(a_7), 7)

# Define the truths
labels = [*range(0, 10)]


guesses = [closest(test) for test in stacks_test[1]]
tensor([guessed_right(guess, labels[1]) for guess in guesses]).float().mean()

# Need to initialise an empty list and then add elements to it, because it's a for loop
res = []
for many_tests, label in zip(stacks_test, labels):
    guesses = [closest(test) for test in many_tests]
    res += [tensor([guessed_right(guess, label)
                   for guess in guesses]).float().mean()]


res
tensor(res).float().mean()
# Success!
# Really good at predicting 1s, and on average about 82% accurate
