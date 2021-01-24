# Neural-Networks-and-Deep-Learning-Models

**Tools Used**
- Python
- [TensorFlow](https://playground.tensorflow.org)  
```
# Installs latest version of TensorFlow 2.X 
pip install --upgrade tensorflow
```

A Neural Network is a powerful machine learning technique that is modelled after neurons in the brain. Neural networks can rival the performance of the most robust statistical algorithms without having to worry about *any* statistical theory. Because of this, neural networks are an in-demand skill for any data scientist. Big tech companies use an advanced form of neural networks called **deep neural networks** to analyze images and natural language processing datasets. Retailers like Amazon and Apple are using neural networks to classify their consumers to provide targeted marketing as well as behavior training for robotic control systems. Due to the ease of implementation, neural networks also can be used by small businesses and for personal use to make more cost-effective decisions on investing and purchasing business materials. Neural networks are scalable and effective—it is no wonder why they are so popular.In this repository we will explore how neural networks are designed and how effective they can be using the **TensorFlow** platform in Python. With neural networks wer can combine the performance of multiple statistical and machine learning models with minimal effort. In fact, more time is spent preprocessing the data to be compatible with the model than spent coding the neural network model, which can be just a few lines of code.  

In this project we work with a mock company, Alphabet Soup, a foundation dedicated to supporting organizations that protect the environment, improve people's well-being, and unify the world. This company has raised and donated a great sum of money to invest in life saving technologies and organized re-forestation groups around the world. Our task will be to analyze the impact of each donation and vet potential recepients. This helps ensure that the foundation's money is being used effectively. Unfortunately, not every dollar the foundation donates is impactful. Sometimes another organization may recieve funds and disapear. As a result, we must work as data scientists to predict which organizations are worth donating to and which are too high risk. This problem seems too complex for statistical and machine learning models we have used. Instead, we will design and train a deep neural network which will evaluate all types of input data and produce a clear decision making result.

## AlphabetSoupCharity Report  
The program for the following report can be found [here](https://github.com/sfnxboy/Deep_Learning_and_Neural_Networks/blob/main/AlphabetSoupCharity/AlphabetSoupCharity.ipynb).  

### Overview of Analysis  
With a [CSV](https://github.com/sfnxboy/Deep_Learning_and_Neural_Networks/blob/main/AlphabetSoupCharity/Resources/charity_data.csv) containing more than 34,000 organizations that the mock company, Alphabet Soup, has recieved over the years, we will attempt to build a neural network model that can accurately predict what features (or characteristics) organizations have that indicate that they will be a succesful investment. Within this dataset there are a number of columns that capture the metadata about each organization, such as the following:  

- **EIN** and **NAME**—Identification columns  
- **APPLICATION_TYPE**—Alphabet Soup application type  
- **AFFILIATION**—Affiliated sector of industry  
- **CLASSIFICATION**—Government organization classification  
- **USE_CASE**—Use case for funding  
- **ORGANIZATION**—Organization type  
- **STATUS**—Active status  
- **INCOME_AMT**—Income classification  
- **SPECIAL_CONSIDERATIONS**—Special consideration for application  
- **ASK_AMT**—Funding amount requested  
- **IS_SUCCESSFUL**—Was the money used effectively  

This project consists of three portions. First we will have to preprocess the data appropriately to be used in a neural network model. Secondly, we will compile, train, and evaluate the model. Lastly, we will attempt to optimize the neural network to improve its performance while being warry of any overfitting that may occur.

### Results  
#### Data Preprocessing  

Our objective is to build a neural network model that can predict with a reasonable degree of accuracy which organizations are most likely to succeed. Given a dataset with 11 features, we can start off by removing columns that have nothing the machine may find meaningful, such as the `EIN` and `NAME` columns. The `IS_SUCCESFUL` column will be our target feature for our neural network model. Thankfully the values in that column are already numeric (values are either 1 or 0), so we will not have to encode that column.  

Still, part of the preprocessing process is encoding categorical variables into numeric values so that the machine may find meaning in those features. Sometimes a column has so many unique values, that it may be best to consider 'bucketing' the infrequent occurences into a single variable. I decide to bucket application types that appear less than 250 times into their own category. I made my decision based on the results from the `value_count()` and `.plot.density()` methods. To complete my preprocessing step I create a OneHotEncoder instance to quantify all categorical features.  

#### Compiling, Training, and Evaluating the Model  

I built my first model with 2 hidden layers, with 80 and 30 nodes respectively. The resulted in the following performance score: `loss: 0.5537 - accuracy: 0.7258`. Considering I only ran the first model with 50 epochs, I decided to change the number of epochs and only the number of epochs in the second model. This revision resulted in the following: `loss: 0.5560 - accuracy: 0.7261`. This is an insignificant difference. My next step was to add another hidden later. My hidden layers were programmed as follows:  
```
# First hidden layer
nn_new1.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1,
    input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn_new1.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Third hidden layer
nn_new1.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="sigmoid"))

# Output layer
nn_new1.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
```  
This model had the following performance: `loss: 0.5557 - accuracy: 0.7255`. Again, an insignificant change. Adding a third hidden layer did not change the performance of the model by much. In my fourth attempt to build a model I changed the activation function of the second hidden layer to a sigmoid function, the results are as follows: `loss: 0.5526 - accuracy: 0.7258`.

### Summary  
A rule of thumb is that if your test accuracy score is larger than your training accuracy score, there is a chance that your model promotes over fitting. This happens to be the case for all four models I built and ran. Nonetheless, my performance metrics could not reach a standard I was satisfied with, albiet all models were likely overfit. This may be the case because there is a noisy feature that is confusing the machine, or because the data itself does not contain enough information for the machine. 
