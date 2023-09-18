# FITC image prediction using cGAN

## Data 
The data consists of pairs of neutrophil phase contrast image and FITC image, in which the FITC intensity should monotonicly related to intracellular ROS concentration by design of the experiment. The experimental pipeline and the image preprocessing steps are in  [image-preprocessing-pipeline](https://github.com/weiliao97/image-preprocessing-pipeline). 

Here are examples of the data with the quantified ROS value labeled.

![phase_fitc](/images/phase_fitc_ros.png "phase_fitc")

## Model 

The goal is given the phase contrast image, the model learns to generate the FITC image. 

![task](/images/task.png "task")

The model consists of a generator and a discriminator and their architecture are as follows: 

![gen](/images/gen.png "gen")

![dis](/images/dis.png "dis")

## Results

The generation results are demonstrated here: 

![results](/images/results.png "results")

