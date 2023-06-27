# Column Normalizer

This repository contains the implementation of a deep learning model for a text classification task. To replicate the obtained results - 
1. Upload the dataset as data.csv to the repository.
2. Install the necessary dependencies as given in requirements.txt.
3. Run the main.py file. This will clean, preprocess and format the data, fit a fine-tuned pretrained BERT model to our dataset (currently the best performing model), and then return the overall as well as the classwise accuracies.
4. Code for finding the optimal 1D CNN architecture for our model using Neural Architecture Search (NAS) is present in the nas_darts folder (work ongoing). This requires custom character embeddings, which can be obtained by running the file character_embeddings.py.
