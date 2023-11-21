from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
from keras import backend as K
import numpy as np
import keras
import nibabel as nib
import pickle
from nilearn.image.image import check_niimg
from nilearn.image.image import _crop_img_to as crop_img_to
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tifffile import imsave
import os
from radiomics import featureextractor
import pandas as pd
import glob
import csv
import SimpleITK as sitk
import random


def crop_img(img, rtol=1e-8, copy=True, return_slices=False):
    img = check_niimg(img)
    data = img.get_data()
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)
    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    if return_slices:
        return slices

    return crop_img_to(img, slices, copy=copy)


class SubDataGenerator(keras.utils.data_utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=1, dim=(240, 240, 155), n_channels=4,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = None
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y1 = np.empty((self.batch_size, 3, *self.dim))
        y2 = np.empty(self.batch_size)

        # Generate data
        # Decode and load the data
        for i, ID in enumerate(list_IDs_temp):
            # 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT)
            # The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. The appearance of the necrotic (NCR) and the non-enhancing (NET) tumor core is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.
            # The labels in the provided data are:
            # 1 for NCR & NET (necrotic (NCR) and the non-enhancing (NET) tumor core) = TC ("tumor core")
            # 2 for ED ("peritumoral edema")
            # 4 for ET ("enhancing tumor")
            # 0 for everything else

            X[i,] = pickle.load(open("./output/%s_images.pkl" % (ID), "rb"))

        return X


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss

model = load_model('model3_new.h5', custom_objects={'InstanceNormalization': InstanceNormalization(axis=1),
                                                    'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss,
                                                    'weighted_dice_coefficient': weighted_dice_coefficient,
                                                    'dice_coefficient': dice_coefficient,
                                                    })
params = {'dim': (160, 192, 160),
          'batch_size': 1,
          'n_classes': 3,
          'n_channels': 4,
          'shuffle': False}


def cropforprediction(f1, f2, f3, f4):
    newimage = nib.concat_images([f1, f2, f3, f4])
    cropped = crop_img(newimage)
    img_array = np.array(cropped.dataobj)
    z = np.rollaxis(img_array, 3, 0)

    padded_image = np.zeros((4, 160, 192, 160))
    padded_image[:z.shape[0], :z.shape[1], :z.shape[2], :z.shape[3]] = z

    a, b, c, d = np.split(padded_image, 4, axis=0)

    images = np.concatenate([a, b, c, d], axis=0)

    ID = 'test'
    pickle.dump(images, open("./output/%s_images.pkl" % (ID), "wb"))
    print('cropped')


# from tkinter.messagebox import showinfo


# Let's create the Tkinter window


window = Tk()
window.title("Overall Survival Predictin")
window.geometry("605x600")
window['background'] = 'lightgrey'

# -------------------------------------
# =============== Style Config =============
# --------------------------------------

# Create style Object
style = Style()

# Will add style to every available button
# even though we are not passing style
# to every button widget.
style.configure('TButton', font=
('calibri', 13, 'bold'),
                foreground='black')

# -------------------------------------
# --------------------------------------


# age varaibale
age = IntVar()

# -------------------------------------
# =============== Methods =============
# --------------------------------------

# list of files' paths
names = []


# Method to select files from directory

def openFile():
    global filename
    filename = filedialog.askopenfilename(initialdir="E:\images", title='Open a file', filetypes=filetypes)
    names.append(filename)
    # l1 = tkinter.Label(window, text = "File path: " + filename)
    # showinfo(title='Selected Files',message=filename)


# Method to get age input from input field

def getvalue():
    # print("Entered Age : ", age.get())
    patient_age = age.get()
    print("Entered Age : ", patient_age)
    print()


# method to print files' paths

def getFeatures(predictions_1_pred):
    ID = 'test'
    paramPath = os.path.join(os.getcwd(), "Params.yaml")
    extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
    image = pickle.load(open("./output/%s_images.pkl" % (ID), "rb"))
    keys, values = [], []
    modalities = [0, 2]
    for i in modalities:
        for j in range(3):
            image = sitk.GetImageFromArray(image[i])
            label = sitk.GetImageFromArray(predictions_1_pred[0][0][j])
            result = extractor.execute(image, label)
            for key, value in result.items():
                keys.append(key)
                values.append(value)
            if (i == 2 & j == 2):
                keys.append('Age')
                values.append(age.get())

            image = pickle.load(open("./output/%s_images.pkl" % (ID), "rb"))
    with open(f"./output/{ID}_features.csv", "w") as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(keys)
        csvwriter.writerow(values)
        print('features ectracted')


dic = {'i': 0}


def predict():
    global l1
    global l

    if dic.get('i') == 1:
        l = Label(window, text='Processing....', background="lightgrey", font='Helvetica 12 bold')
        l.place(x=180, y=350)
        l1 = Label(window, text='Please Wait', background="lightgrey",font='Helvetica 12 bold')
        l1.place(x=180, y=380)
    print(dic)
    ID = 'test'
    cropforprediction(names[0], names[1], names[2], names[3])
    validation_generator = SubDataGenerator(['test'], **params)
    predictions_1_pred = model.predict_generator(generator=validation_generator)
    X = pickle.load(open("./output/%s_images.pkl" % (ID), "rb"))
    for i in range(4):
        for j in range(3):
            imarray = predictions_1_pred[0][0][j]
            imarray = (imarray * 255.0)
            imarray = imarray + X[i]
            nib_img = nib.Nifti1Image(imarray, affine=np.eye(4))
            nib.save(nib_img, f'./output/test_predicted_m{i}_c{j}.nii.gz')
    print('file saved')
    names.clear()
    getFeatures(predictions_1_pred)
    x = pd.read_csv(f"./output/test_features.csv")
    data = pd.read_csv('./features/from_predictions/top_corr_data_20_c.csv')
    del data['Survival']
    s_model = pickle.load(open('survival_model', 'rb'))
    survival = s_model.predict(x[data.columns])
    print(survival)
    if survival == 0:
        l = Label(window, text='Patient is a short Survivor', background="lightgrey", font='Helvetica 12 bold')
        l.place(x=180, y=350)
        l1 = Label(window, text='files have been saved in output folder', background="lightgrey",
                   font='Helvetica 12 bold')
        l1.place(x=180, y=380)
        dic['i'] = 1
        # img = Image.open("short.png")
        # img = img.resize((100, 100), Image.ANTIALIAS)
        # img = ImageTk.PhotoImage(img)
        # panel = Label(window, image=img)
        # panel.image = img
        # panel.place(x=250, y=390)

    elif survival == 1:
        l = Label(window, text='Patient is a mid Survivor', background="lightgrey", font='Helvetica 12 bold')
        l.place(x=180, y=350)
        l1 = Label(window, text='files have been saved in output folder', background="lightgrey",
                   font='Helvetica 12 bold')
        l1.place(x=180, y=380)
        dic['i'] = 1
        # img = Image.open("mid.png")
        # img = img.resize((100, 100), Image.ANTIALIAS)
        # img = ImageTk.PhotoImage(img)
        # panel = Label(window, image=img)
        # panel.image = img
        # panel.place(x=250, y=390)

    elif survival == 2:
        l = Label(window, text='Patient is a long Survivor', background="lightgrey", font='Helvetica 12 bold')
        l.place(x=180, y=350)
        l1 = Label(window, text='files have been saved in output folder', background="lightgrey",
                   font='Helvetica 12 bold')
        l1.place(x=180, y=380)
        dic['i'] = 1
        # img = Image.open("long.png")
        # img = img.resize((100, 100), Image.ANTIALIAS)
        # img = ImageTk.PhotoImage(img)
        # panel = Label(window, image=img)
        # panel.image = img
        # panel.place(x=250, y=390)



# ---------------------------------------
# ============ Tkinter widgets =========
# ---------------------------------------


lbl1 = Label(window, text="Survival Prediction", background="lightgrey", font='Helvetica 20 bold').place(x=160, y=10)

# You will create input field for age variable

lbl2 = Label(window, text="Age", background="lightgrey", font=(None, 15)).place(x=110,
                                                                                y=86)  # 'Age' label is placed at x = 15 and y = 30

# 'Entry' class is used to display the input-field for 'Age' text label
ent = Entry(window, textvariable=age, font=(None, 12)).place(x=170, y=85,
                                                             height=27)  # first input-field is placed at x= 60, y=30

filetypes = (
    ('nifti files', '*.nii.gz'),
    ('All files', '*.*')
)

# Button for 1st Modality
btn1 = Button(window, text="Choose Flair", command=openFile, style='TButton').place(x=10, y=170)

# Button for 2nd Modality
btn2 = Button(window, text="Choose T1", command=openFile, style='TButton').place(x=160, y=170)

# Button for 3rd Modality
btn3 = Button(window, text="Choose T1CE", command=openFile, style='TButton').place(x=310, y=170)

# Button for 4th Modality
btn4 = Button(window, text="Choose T2", command=openFile, style='TButton').place(x=460, y=170)

# Button for Prediction
btn6 = Button(window, text="Predict Results", command=predict, style='TButton').place(x=230, y=300)

window.resizable(False, False)
window.mainloop()
