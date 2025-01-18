Brief Summary of the Project:
Dataset Preparation: -Loaded and preprocessed image data for training and validation. -Normalized pixel values (rescaling to [0, 1]) for better training efficiency.
Model Building: -Created a Convolutional Neural Network (CNN) using the Sequential API. -Added Convolutional, MaxPooling, Flatten, Dropout, and Dense layers to extract features and classify images.
Model Compilation: -Adam optimizer was used for efficient gradient descent. -Chose SparseCategoricalCrossentropy as the loss function for multi-class classification. -Tracked accuracy as the performance metric.
Model Training: -Trained the model over 25 epochs using the fit function. -Monitored training and validation accuracy/loss during each epoch.
Evaluation & Visualization: -Plotted training/validation accuracy and loss to assess the model’s performance.
Prediction: -Used the trained model to predict the category of unseen images and displayed the accuracy score. Images randomnly selected from test set and/or google images.
