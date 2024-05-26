#This function is used in Notebook 1: 2.1. Train and Test Folders
import os
def count_images_in_folder(folder_path):
    '''
    Counts the number of images in a specified folder.

    Parameters:
    folder_path (str): The path to the folder containing the images.

    Returns:
    int: The count of images present in the folder.
    '''
    images = os.listdir(folder_path)
    image_count = len(images)
    return image_count


#This function is used in Notebook 1: 2.2.3. Visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def plot_count_with_annotations(data, column, figsize=(6, 4)):
    '''
    Plot the distribution of classes in a specified column of a DataFrame.

    Parameters:
    data (DataFrame): The DataFrame containing the data to plot.
    column (str): The column in the DataFrame for which the distribution will be plotted.
    figsize (tuple, optional): The size of the figure (width, height). Default is (6, 4).

    Returns:
    None (displays the plot)
    '''
    plt.figure(figsize=figsize)
    ax = sns.countplot(data=data, x=column)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.title(f'Distribution of {column} Classes')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.show()

    
#This function is used in Notebook 1: 2.2.6. Encoding Labels 
import pandas as pd

def encode_categorical_features(dataframe, categorical_columns):
    """
    Encode categorical columns using a combination of label encoding and one-hot encoding.

    Parameters:
    - dataframe (DataFrame): The input DataFrame containing categorical columns.
    - categorical_columns (list): A list of column names to be encoded.

    Returns:
    - encoded_dataframe (DataFrame): DataFrame with encoded categorical columns.
    """
    one_hot_columns = []

    for column in categorical_columns:
        if column == 'dx':
            # Create a new column 'dx_coded' with pd.Categorical().codes for 'dx' column
            dataframe['dx_coded'] = pd.Categorical(dataframe['dx']).codes
        elif column == 'age':
            # Create a new column 'age_coded' with pd.Categorical().codes for 'age' column
            dataframe['age_coded'] = pd.Categorical(dataframe['age']).codes
            dataframe.drop('age', axis=1, inplace=True)  # Drop the original 'age' column
        else:
            # Add other categorical columns for one-hot encoding
            one_hot_columns.append(column)

    # Apply one-hot encoding to the selected columns
    encoded_dataframe = pd.get_dummies(dataframe, columns=one_hot_columns, drop_first=True)

    return encoded_dataframe


#This function is used in Notebook 1: 3. Adding the path of the images to MetaData
import os
import glob
import pandas as pd

def load_and_merge_df(df_encoded, folder):
    """
    Loads image paths from a specified folder and merges them with the encoded DataFrame.

    Args:
    - df_encoded (pd.DataFrame): DataFrame with categorical columns encoded
    - folder (str): Path to the folder containing image files

    Returns:
    - pd.DataFrame: Merged DataFrame containing image paths and encoded categorical data
    """
    # Load image paths
    image_paths = []
    for file in glob.glob(os.path.join(folder, '*.jpg')):
        image_paths.append(file)

    # Create DataFrame with image paths
    df = pd.DataFrame(image_paths, columns=['path'])
    df['image_id'] = df['path'].apply(lambda x: os.path.basename(x).split('.')[0])

    # Merge with df_encoded
    merged_df = df.merge(df_encoded, on='image_id', how='inner')

    return merged_df


#This function is used in Notebook 1: 4. Train Test Split 
import matplotlib.pyplot as plt

def plot_cancer_distribution(dataframe, title):
    """
    Plots the percentage distribution of cancer types(dx) in a DataFrame.

    Parameters:
    dataframe (DataFrame): The DataFrame containing cancer information.
    title (str): Title for the plot.

    Returns:
    None (displays the plot).
    """
    # Calculate value counts for 'dx' column
    cancer_distribution = dataframe['dx'].value_counts()

    # Calculate percentage distribution
    percentage_cancer_distribution = (cancer_distribution / len(dataframe)) * 100

    # Create a pie chart
    plt.figure(figsize=(6, 6))
    percentage_cancer_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=140)
    plt.title(f'Percentage Distribution of Cancer Types - {title}')
    plt.ylabel('')
    plt.show()

    

#This function is used in Notebook 1: 5.1. Visualization  
import matplotlib.pyplot as plt
from PIL import Image

def plot_image_samples(dataframe, num_images_per_dx):
    """
    Plot sample images from each diagnosis category in the DataFrame.

    Parameters:
    - dataframe (DataFrame): DataFrame containing image paths and diagnosis labels.
    - num_images_per_dx (int): Number of sample images to plot for each diagnosis.
    - filename (str): Optional filename to save the plot. Defaults to None.

    Returns:
    - None (displays the plot or saves it as an image file)
    """
    # Get unique diagnosis categories
    unique_dx = dataframe['dx'].unique()

    # Create subplots for each diagnosis category
    fig, axes = plt.subplots(len(unique_dx), num_images_per_dx, figsize=(3 * num_images_per_dx, 3 * len(unique_dx)))

    for i, dx_type in enumerate(unique_dx):
        # Filter dataframe for each diagnosis category
        filtered_df = dataframe[dataframe['dx'] == dx_type]
        # Randomly select sample images for each category
        image_samples = filtered_df.sample(n=num_images_per_dx)

        for j in range(num_images_per_dx):
            image_path = image_samples.iloc[j]['path']
            img = Image.open(image_path)

            # Set the subplot for the image
            ax = axes[i, j]
            ax.imshow(img)

            # Set title with diagnosis label and code
            title_text = f"{image_samples.iloc[j]['dx']} - {image_samples.iloc[j]['dx_coded']}"
            ax.set_title(title_text)

            ax.axis('off')

    plt.tight_layout()
    plt.show

    
#This function is used in Notebook 1: 5.2. Hair Removal 
def process_images(dataframe):
    """
    Process images in the given DataFrame to remove hair.

    Parameters:
    - dataframe (DataFrame): Input DataFrame containing image paths.

    Notes:
    This function iterates through each row of the DataFrame, processes the images to remove hair,
    and stores the processed images in a new column 'image_no_hair'.

    Requirements:
    - 'remove_hair' function or processing logic for hair removal.

    """
    dataframe['image_no_hair'] = None

    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        image_path = row['path']
        img = Image.open(image_path)
        img_array = np.array(img)
        processed_images = remove_hair([img_array])

        if processed_images:  # Check if the list is not empty
            dataframe.at[index, 'image_no_hair'] = processed_images[0]
            
            
#This function is used in Notebook 1: 5.2. Hair Removal
import matplotlib.pyplot as plt

def plot_images_with_hair_removed(dataframe, num_images=5):
    """
    Plot original images alongside images with hair removed from a DataFrame.

    Args:
    - dataframe (pandas.DataFrame): The DataFrame containing image paths and images with hair removed.
    - num_images (int): Number of images to display. Default is 5.

    Displays:
    - Matplotlib plot: Grid of original images and images with hair removed.
    """
    images_with_hair = dataframe['image_no_hair']
    original_images = dataframe['path']

    fig, axes = plt.subplots(num_images, 2, figsize=(6, 2 * num_images))
    for i in range(num_images):
        img = plt.imread(original_images.iloc[i])
        hair_removed_img = images_with_hair.iloc[i]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(hair_removed_img)
        axes[i, 1].set_title('Image with Hair Removed')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

#This function is used in Notebook 1: Crop the Border
import matplotlib.pyplot as plt

def plot_images_with_hair_removed_cropped(dataframe, num_images=5):
    """
    Plot images with hair removed alongside their cropped versions from a DataFrame.

    Args:
    - dataframe (pandas.DataFrame): The DataFrame containing images with hair removed and their cropped versions.
    - num_images (int): Number of images to display. Default is 5.

    Displays:
    - Matplotlib plot: Grid of images with hair removed and their cropped versions.
    """
    images_with_hair = dataframe['image_no_hair']
    images_cropped = dataframe['cropped_images']

    fig, axes = plt.subplots(num_images, 2, figsize=(6, 2 * num_images))
    for i in range(num_images):
        hair_removed_img = images_with_hair.iloc[i]
        hair_removed_img_cropped = images_cropped.iloc[i]

        axes[i, 0].imshow(hair_removed_img)
        axes[i, 0].set_title('Image with Hair Removed')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(hair_removed_img_cropped)
        axes[i, 1].set_title('Cropped Image')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
    
#This function is used in Notebook 1: 5.4. Identification of Colors
import matplotlib.pyplot as plt

def plot_color_samples(dataframe, color_label, num_samples=20, rows=4, cols=5, figsize=(15, 12)):
    """
    Plots sampled images from a DataFrame based on a specific color label.

    Args:
    - dataframe (pandas.DataFrame): The DataFrame containing image data.
    - color_label (str): The column name representing the color label to filter images.
    - num_samples (int): Number of images to sample and display. Default is 20.
    - rows (int): Number of rows in the grid layout. Default is 4.
    - cols (int): Number of columns in the grid layout. Default is 5.
    - figsize (tuple): Figure size for the plot. Default is (15, 12).

    Displays:
    - Matplotlib plot: Grid of sampled images based on the specified color label.
    """
    color_rows = dataframe[dataframe[color_label] == 1].sample(n=num_samples)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, (index, row) in enumerate(color_rows.iterrows()):
        image = row['cropped_images']

        ax = axes[i // cols, i % cols]
        ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#This function is used in Notebook 1: 5.5. Removing Non Hematoma Area 
import matplotlib.pyplot as plt

def plot_random_images(dataframe, num_images):
    """
    Plot a random selection of images from a DataFrame.

    Args:
    - dataframe (pandas.DataFrame): DataFrame containing image data.
    - num_images (int): Number of random images to display.

    Displays:
    - Matplotlib plot: Grid of randomly sampled 'image_no_hair', 'cropped_images', and 'final_preproc_image'.
    """
    random_subset = dataframe.sample(num_images)

    num_columns = 3  # Number of columns for each image type
    num_rows = num_images

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(10, num_rows * 5))

    for i, (_, row) in enumerate(random_subset.iterrows()):
        image_no_hair = row['image_no_hair']
        cropped_image = row['cropped_images']
        final_preproc_image = row['final_preproc_image']

        axes[i, 0].imshow(image_no_hair)
        axes[i, 0].set_title(f"image_no_hair")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(cropped_image)
        axes[i, 1].set_title(f"cropped_images")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(final_preproc_image)
        axes[i, 2].set_title(f"final_preproc_image")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


#This function is used in notebook 2, notebook 3 and notebook 4
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_classification_report_with_matrix(model, X_val, y_val, matrix=True):
    """
    Evaluates a classification model using the validation set and generates a classification report.

    Parameters:
    model (object): The classification model to evaluate.
    X_val (array-like): Validation set features.
    y_val (array-like): Validation set labels.
    matrix (bool): If True, displays the confusion matrix. Defaults to True.

    Returns:
    None (prints the classification report and displays the confusion matrix if matrix=True).
    """
    # Perform predictions on the validation set
    y_pred = model.predict(X_val)

    # Convert predicted probabilities to class labels using argmax
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Convert one-hot encoded y_val to integer labels
    y_true = np.argmax(y_val, axis=1)

    # Calculate and print classification report
    report = classification_report(y_true, y_pred_classes)
    print("Classification Report:")
    print(report)

    if matrix:
        # Calculate and plot confusion matrix
        # Define a mapping dictionary from encoded values to diagnostic labels
        encoded_to_dx = {
            0: 'akiec',
            1: 'bcc',
            2: 'bkl',
            3: 'df',
            4: 'mel',
            5: 'nv',
            6: 'vasc'
        }

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)

        # Convert confusion matrix values to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Calculate percentages

        # Create a heatmap of the confusion matrix with labels as percentages
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_percent, annot=True, cmap='mako_r', fmt='.2f',
                    xticklabels=list(encoded_to_dx.values()),
                    yticklabels=list(encoded_to_dx.values()))

        # Add labels as a legend on the side
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (%)')
        plt.show()


    
    
#This function is used in notebook 2, notebook 3 and notebook 4
import matplotlib.pyplot as plt

def plot_loss(history):
    """
    Plot the training and validation loss over epochs.

    Parameters:
    - history: History object returned by model.fit() containing training/validation loss values.

    Returns:
    - None (displays a plot of training and validation loss)
    """
    plt.figure(figsize=(7, 2))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()



#This function is used in the end of Notebook 2, Notebook 3 and Notebook 4
import numpy as np
from sklearn.metrics import f1_score

def get_f1_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Calculate the weighted F1 scores for a given model's predictions on training, validation, and test sets.

    Parameters:
    - model: Machine learning model used for predictions.
    - X_train: Input features of the training set.
    - X_val: Input features of the validation set.
    - X_test: Input features of the test set.
    - y_train: True labels of the training set.
    - y_val: True labels of the validation set.
    - y_test: True labels of the test set.

    Returns:
    - f1_train: Weighted F1 score on the training set predictions.
    - f1_val: Weighted F1 score on the validation set predictions.
    - f1_test: Weighted F1 score on the test set predictions.
    """
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    y_pred_train_classes = np.argmax(y_pred_train, axis=1)
    y_true_train = np.argmax(y_train, axis=1)

    y_pred_val_classes = np.argmax(y_pred_val, axis=1)
    y_true_val = np.argmax(y_val, axis=1)

    y_pred_test_classes = np.argmax(y_pred_test, axis=1)
    y_true_test = np.argmax(y_test, axis=1)

    f1_train = f1_score(y_true_train, y_pred_train_classes, average='weighted')
    f1_val = f1_score(y_true_val, y_pred_val_classes, average='weighted')
    f1_test = f1_score(y_true_test, y_pred_test_classes, average='weighted')

    return f1_train, f1_val, f1_test

