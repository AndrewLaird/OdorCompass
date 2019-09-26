
<center>  
<h2>Passive Multiplexed Odor Compass for Identifying Multiple Odor  
Sources Using SERS Sensors and Machine Learning</h2>  
</center>
<center>
<p>William John Thrift, Antony Cabuslay, Andrew Benjamin Laird, Saba Ranjbar, Allon I. Hochbaum, and Regina Ragan</p>
</center>

Requirements:
- Tensorflow:  1.14.0+
- Keras: 2.2.4+
- Numpy: 1.17.2+
- Pandas: 0.25.1+
- Matplotlib: 3.1.0+
<h2>
Abstract
</h2>
<p>
Olfaction is important for identifying and avoiding toxic substances in living systems. Many efforts have been made to realize artificial olfaction systems that reflect the capacity of biological systems. A sophisticated example of an artificial olfaction device is the odor compass which uses chemical sensor data to identify odor source direction. Successful odor compass designs often rely on plume-based detection and mobile robots, where active, mechanical motion of the sensor platform is employed. Passive, diffusion-based odor compasses remain elusive as detection of low analyte concentrations and quantification of small concentration gradients from within the sensor platform are necessary. Further, simultaneously identifying multiple odor sources using an odor compass remains an ongoing challenge, especially for similar analytes. Here, we show that surface-enhanced Raman scattering (SERS) sensors overcome these challenges, and we present the first SERS odor compass. Using a grid array of SERS sensors, machine learning analysis enables reliable identification of multiple odor sources arising from diffusion of analytes from one or two localized sources. Specifically, convolutional neural network and support vector machine classifier models achieve over 90% accuracy for a multiple odor source problem. This system is then used to identify the location of an _Escherichia coli_ biofilm via its complex signature of volatile organic compounds. Thus, the fabricated SERS chemical sensors have the needed limit of detection and quantification for diffusion-based odor compasses. Solving the multiple odor source problem with a passive platform opens a path toward an Internet of things approach to monitor toxic gases and indoor pathogens.
</p>

<h2>
How to Use
</h2>
<p>Clone the repository with: `git clone [repository link]`</p>
<p> We have included some of our preprosessed data in the raman_data folder, first, unzip any zipped files in that folder </p>

<p> Then run ModelsOnData.py which will take the data from raman_data and train and report on three types of models:</p>

- Convolutional Neural Networks (CNN) which scans over the data with a window and keeps the spacial information in tact
- Support Vector Classifier (SVC) which uses a flattened version of the input data, and treats that as a vector in a high dimensional plane. The SVC then finds planes that seperate the data into each classification.
- Artificial Neural Network (ANN)which uses a flattened version of the input data, and employs common neural network techniques like dropout.

<p> For the purpose of demonstration we have lowered the number of epochs to train the CNN and ANN so results from running the python file will finish in under an hour but may not match the papers results. If you possess the resources to run for longer, increase the EPOCHS_TO_TRAIN variable in ModelsOnData.py</p>

