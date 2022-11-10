from pickletools import optimize
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

class_names = ['Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

model = models.load_model('image_classifier71.model')

def classify(file_path):
    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32))

    plt.imshow(img, cmap=plt.cm.binary)

    prediction = model.predict(np.array([img]) / 255.0)

    index = np.argmax(prediction)
    label.configure(foreground='#FFFFFF',text=class_names[index], font=('arial',20,'bold'))
    print(f'\n\nPrediction is : {class_names[index]}')


top=tk.Tk()
top.geometry('800x600')
top.title('Image Classification')
top.configure(background='#202325')
label=Label(top,background='#202325', font=('arial',15,'bold'))
sign_image = Label(top)

def show_classify_button(file_path):
    classify_b=Button(top,text="CLASSIFY",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.pack(side=RIGHT, padx=80, pady=90)
    
def upload_image(): 
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="UPLOAD",command=upload_image,padx=10,pady=8)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=RIGHT, padx=80, pady=80)
sign_image.pack(side=LEFT,expand=True)
label.pack(side=BOTTOM,expand=True)

heading = Label(top, text="KNOW YOUR IMAGE", font=('MT Bold',30,'bold'))
heading.configure(background='#202325',foreground='#FFFFFF')
heading.pack(side=TOP)
top.mainloop()