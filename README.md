# Face Segmentation

This project parses different parts of the face using semantic segmentation. The Machine
learning model used is U-Net.    
The experiments folder contains application of semantic segmentation i.e. to change lip
and hair color. Refer [Github](https://github.com/Sanjana7395/face_makeup_app.git) code 
for browser application to do the same using tensorflow.js and opencv.js.

Configuration of Project Environment
=====================================

1. Clone the project.
2. Install packages required.
3. Download the data set
4. Run the project.

Setup procedure
----------------
1. Clone project from [GitHub](https://github.com/Sanjana7395/face_segmentation.git).  
      Change to the directory face_segmentation.
2. Install packages  
   In order to reproduce the code install the packages 
   
   1. Manually install packages mentioned in requirements.txt file or use the command.

           pip install -r requirements.txt

   2. Install packages using setup.py file.

            python setup.py install

   The **--user** option directs setup.py to install the package
   in the user site-packages directory for the running Python.
   Alternatively, you can use the **--home** or **--prefix** option to install
   your package in a different location (where you have the necessary permissions)

3. Download the required data set.  
      The data set that is used in this project CelebAMask-HQ that is available
      [here](https://github.com/switchablenorms/CelebAMask-HQ).

4. Run the project.  
      See **Documentation for the code** section for further details.
      
Documentation for the code
===========================

1. __Pre processing__  
   This folder contains  
      
   1. Code to generate mask from the different label definitions given in the dataset and split the data
   into train, validation and test set. This is present in preprocessing/load_data.py. 
   To execute this code, within the 'preprocessing' folder enter the below
   command
           
           python load_data.py
              
   2. Augment data. The code is present in preproprocessing/augment_dataset.py.

2. __Models__  
   This folder contains the model used in this project namely, U-Net

3. __train.py__ 
   
   Run the code using the below command 
                    
         python train.py -m <model_name>
          
   For help on available models
   
         python train.py --help

4. __test.py__  
    This file helps in visualizing segmentation for a given test image. Usage is as follows
      
         python test.py -v <visualization_method>
         
      for help on usage
      
         python app.py --help

5. __experiments__    
    This folder contains the code to change the lip and hair color from the segmentation mask obtained.
      
Results
========

Below are the results obtained on the test set for the models trained in the project.

> NOTE    
   The results obtained are system specific. Due to different combinations of the neural 
   network cudnn library versions and NVIDIA driver library versions, the results can be 
   slightly different. To the best of my knowledge, upon reproducing the environment, the
   ballpark number will be close to the results obtained.

| Models                           | Accuracy (%)  | mIoU (%)  |
|----------------------------------|:-------------:|:---------:|
| U Net                            | 93.13         | 60.90     |
