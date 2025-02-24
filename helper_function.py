import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import zipfile
import datetime
import tensorflow as tf
import requests
     

def plot_image(target_folder,target_class):
  """
  returns and plot a random image from the given directory

  Args
  targert_folder : food_vision/test
  target_class : spaghetti_carbonara

  example use of the function
  plot_image(target_folder="food_vision/test",target_class="spaghetti_carbonara")

  """
  target_file = os.path.join(target_folder,target_class)
  random_image = random.sample(os.listdir(target_file),1)
  img = mpimg.imread(os.path.join(target_file,random_image[0]))

  plt.figure(figsize=(15,15))
  plt.imshow(img)
  plt.axis("off")

  return img

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
     

def plot_loss_curve(history):

  """
  plots the training accuracy, training loss and validation accuracy and validation loss

  example use of the function
  plot_loss_curve(model_history)

  """
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']
  epochs = range(len(history.history['accuracy']))

  plt.plot(epochs,loss,label="training_loss")
  plt.plot(epochs,val_loss,label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  plt.figure()
  plt.plot(epochs,accuracy,label="training_accuracy")
  plt.plot(epochs,val_accuracy,label="val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()

     

def walk_through_dir(dir_path):

  """
  walks through the all directories given folder path

  """
  for dirpath,dirnames,filenames in os.walk(dir_path):
    print(f"there are {len(dirnames)} directories in {dir_path}")
     

def list_down_filenames(target_folder):

  """
  listdown all the filenames inside a folder

  """
  file_names = os.listdir(target_folder)
  print(f"there are {len(file_names)}")
  print(file_names)
     

def unzip_data(zip_name):

  """
  unzips the files in a zip file read only
  """
  zip_ref = zipfile.ZipFile(zip_name,"r")
  zip_ref.extractall()
  zip_ref.close()

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


def load_image_from_a_location(target_location,is_network=False,image_size=224,is_scaled=False):
  """
  returns a resized image from a given location

  Args
  target_location : network or local path to the image
  image_size : specify the expected resize of the image
  is_scaled : normalize the image between 0 and 1

  example use of the function

  load_image_from_a_location(target_location="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f)

  """

  if target_location.startswith("http"):
        # If it's a URL, download the image using requests
    try:
      response = requests.get(target_location, stream=True)
      response.raise_for_status()  # Raise an exception for bad responses
      target_image = tf.image.decode_jpeg(response.content)  # Decode the JPEG content
    except Exception as e:
      print(f"Error downloading image from {target_location}: {e}")
      return None

  else:
    target_image = tf.io.read_file(target_location)
    target_image = tf.image.decode_image(target_image)

        # If it's a local path, use tf.io.read_file

  plt.figure(figsize=(10,10))
  plt.imshow(target_image)
  plt.axis("off")

  target_image = tf.image.resize(target_image,size=[image_size,image_size])

  if is_scaled:
    return target_image/255
  else:
    return target_image
     
