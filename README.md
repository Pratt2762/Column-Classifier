# Column Normalizer

This repository contains the implementation of two deep learning approaches for a text classification task. To replicate the obtained results - 
1. Upload the dataset as data.csv to the parent directory.
2. Install the necessary dependencies as given in requirements.txt.
3. To use BERT for the text classification task using pre-trained word embeddings, run the main.py file in the bert_classification folder.
4. To find out the optimal 1D CNN model for our text classification task using Neural Architecture Search (NAS), first run the create_embeddings.py file in the nas_darts folder. Then run the train.py file present in the same folder. This will create a new folder in the parent directory containing the training and validation logs and the weights of the model obtained by the NAS algorithm.
5. The training and model hyperparameters present in the train.py file should be chosen after manual testing to check which ones give the best results.
6. To view the testing results, run the test.py file.
7. To view the normal and reduction cells obtained by the NAS algorithm in the form of a Direct Acyclic Graph (DAG), run the visualize.py file. The images will be saved in pdf format as normal.pdf and reduction.pdf.
