
# Handwritten Digit Recognition with CNN and FPGA

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The model is trained using PyTorch, and a Pygame interface allows real-time digit drawing and prediction. Predictions are sent via UART to a Basys3 FPGA, where Verilog modules process and display them on a 7-segment display.

## Features
- Train a CNN on the MNIST dataset.
- Real-time digit drawing and prediction using a Pygame interface.
- UART transmission of predictions to an FPGA.
- FPGA-based 7-segment display of recognized digits.

## Project Structure
- `src/`: Contains Python and Verilog scripts
  - `train_cnn.py`: Trains the CNN model on MNIST and saves it as `cnn_model.pth`.
  - `touchscreen_predict.py`: Real-time digit drawing, prediction, and UART output.
  - `uart_rx.v`: Verilog module for UART reception.
  - `seg7_decoder.v`: Verilog module to decode digits for 7-segment display.
  - `uart_display_top.v`: Top-level Verilog module integrating UART and display logic.
- `constr/`: Contains constraint files
  - `basys3_xdc`: Constraints file for Basys3 pin mappings.

