# CBIL-VHPLI

CBIL–VHPLI utilizes zero-padding and cropping tricks to turn protein/RNA sequences of variable lengths into fixed-length sequences, allowing them to meet the input requirements of a CNN model. The high-order one-hot encoding method is used to transform protein/RNA sequences into image-like inputs, capturing dependencies among amino acids or nucleotides. Additionally, protein sequences use CTDD to obtain deeper sequence signatures, while RNA sequences use Z-curve curves to capture deeper dependencies between nucleotides. These encoded sequences are then fed into the convolutional neural network for processing. The output from the convolutional layer is further processed through the Bidirectional Long Short Term Memory Network (BiLSTM) with convolutional filters and maximum pooling layers. After each BiLSTM layer outputs a one-dimensional feature vector, the features are fused and passed through the MLB layer to generate the final prediction results. Finally, transfer learning is applied to the viral protein-lncRNA interaction dataset to predict interactions between lncRNAs and viral proteins.

## System Requirements

CBIL‒VHPLI was developed using the Keras API with TensorFlow as a backend. The following dependencies are required:
- TensorFlow version=2.11.1
- scikit-learn version=1.3.2
- numpy version=1.24.4
- Keras version=2.11.0
- pandas version=2.0.3

## Content

- `CBIL–VHPLI_pretrain.py`: The main Python code to reproduce our pre-trained model weights.
- `CBIL–VHPLI_finetune.py`: Python code for fine-tuning the best pre-trained model weights.
- `data.zip`: Contains training, testing, and independent testing datasets, along with fine-tuning data set with sequence pair names and labels.

## User's Guide
Users must process experimental data according to the specified requirements for training or testing data files.

### Data Processing

Users need to preprocess experimental data as required for training or testing data files. Follow these steps to execute the code:

1. **Set up the environment**:
    - Ensure all dependencies are installed.
    - Unzip `data.zip` and place the data in the appropriate directory.

2. **Train the Model**:
    - Execute `CBIL–VHPLI_pretrain.py` to train the model on the provided datasets.

3. **Testing and Fine-tuning**:
    - Use `CBIL–VHPLI_finetune.py` to fine-tune the model with the best pre-trained weights using the independent testing dataset and the fine-tuning data set.

### Running the Code

To run the code, navigate to the directory containing the scripts and execute the following commands:

```bash
python CBIL–VHPLI_pretrain.py
python CBIL–VHPLI_finetune.py
