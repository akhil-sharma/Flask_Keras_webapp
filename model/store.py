import urllib.request
import os
import zipfile
import sys


url = 'http://www.superdatascience.com/wp-content/uploads/2017/04/Convolutional_Neural_Networks.zip'
last_percent_reported = None
data_root = '.' # Change me to store data elsewhere

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 5% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()
      
    last_percent_reported = percent
        
def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  dest_filename = os.path.join(data_root, filename)
  if force or not os.path.exists(dest_filename):
    print('Attempting to download:', filename) 
    print(url)
    proxy = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy)
    opener.addheaders = [('User-Agent','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30')]
    urllib.request.install_opener(opener)
    filename, _ = urllib.request.urlretrieve(url, dest_filename, reporthook=download_progress_hook)
    print('\nDownload Complete!')
  statinfo = os.stat(dest_filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', dest_filename)
  else:
    raise Exception(
      'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
  return dest_filename


train_filename = maybe_download('Convolutional_Neural_Networks.zip',233354462, force = False)


def maybe_extract(filename, force=False):
  root = os.path.splitext(filename)[0]  # remove .zip
  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    zip_ref = zipfile.ZipFile(filename)
    sys.stdout.flush()
    zip_ref.extractall()

  data_folder = os.path.join(root, 'dataset')
  print(data_folder)
  return data_folder

folder_in_question = maybe_extract(train_filename)
training_set_folder = os.path.join(folder_in_question, "training_set")
test_set_folder = os.path.join(folder_in_question, "test_set")


def build_model(training_set_path, test_set_path, common_batch_size = 32, n_epochs = 25):
	from keras.models import Sequential
	from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
	from keras.preprocessing.image import ImageDataGenerator 

	# Initializing the CNN
	classifier = Sequential()

	# Adding the first convolutional layer
	classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	
	# Adding the second convolutional layer
	classifier.add(Conv2D(32, (3,3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size=(2,2)))
	
	# Adding the third convolutional layer (necessity can be argued upon)
	classifier.add(Conv2D(32, (3,3), activation = 'relu'))
	classifier.add(MaxPooling2D(pool_size = (2,2)))
	
	# Flattening
	classifier.add(Flatten())
	
	# Fully connected 
	classifier.add(Dense(units = 128, activation = 'relu'))
	classifier.add(Dense(units = 1, activation = 'sigmoid'))
	
	# Compiling the CNN
	classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

	#preparing the datasets
	training_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2,horizontal_flip = True)
	training_set = training_datagen.flow_from_directory(training_set_path, target_size = (64, 64),batch_size = common_batch_size,class_mode = 'binary')

	test_datagen = ImageDataGenerator(rescale = 1./255)
	test_set = test_datagen.flow_from_directory(test_set_path,target_size = (64,64),batch_size = common_batch_size,class_mode = 'binary')

	# Fitting the dataset to the classifier
	classifier.fit_generator(training_set,steps_per_epoch = 8000 / common_batch_size,epochs = n_epochs,validation_data = test_set,validation_steps = 2000)

	return classifier


model = build_model(training_set_folder, test_set_folder)

model.save('cat_dog_model.h5')

print("model saved")
