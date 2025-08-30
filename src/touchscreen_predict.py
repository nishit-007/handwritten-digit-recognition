import pygame
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import serial
from PIL import Image


MODEL_PATH = "cnn_model.pth"
UART_PORT = "COM4"   
BAUD_RATE = 9600
SEND_UART = True     


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1   = nn.Linear(20 * 4 * 4, 50)
        self.fc2   = nn.Linear(50, 10)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 20 * 4 * 4)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = torch.load(
    MODEL_PATH,
    map_location=torch.device('cpu'),
    weights_only=False    
)
model.eval()


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


pygame.init()
WIDTH, HEIGHT = 280, 280
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a Digit (Press Enter to Predict, C to Clear)")

canvas = pygame.Surface((WIDTH, HEIGHT))
canvas.fill(BLACK)


if SEND_UART:
    ser = serial.Serial(UART_PORT, BAUD_RATE)

def predict_digit(surface):
  
    pygame.image.save(surface, "temp.png")
    image = Image.open("temp.png")
    image = transform(image).unsqueeze(0)  

    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


running = True
drawing = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

       
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                digit = predict_digit(canvas)
                print(">>> Predicted Digit:", digit)
                if not SEND_UART:
                    continue
                ser.write(bytes([digit]))
                print("Sent to UART:", digit)
            elif event.key == pygame.K_c:
                canvas.fill(BLACK)

    if drawing:
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(canvas, WHITE, mouse_pos, 8)

    screen.blit(canvas, (0, 0))
    pygame.display.flip()

pygame.quit()

