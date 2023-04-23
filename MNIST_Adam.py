import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import clear_output

#Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() 


#Preprocess the data
x_train = np.reshape(x_train, (60000,784))/255.0
x_test = np.reshape(x_test, (10000,784))/255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])

##Print the input shape
print(x_train.shape)
#So we have 784 nodes (for 28x28 pixels)

#Print the output shape
print(y_train.shape)
#10 outputs nodes represent for digital number range from 0 to 10

#Define function
def relu(x):
    return np.maximum(x,0)
def sigmoid(x):
    return 1./(1.+np.exp(-x))
def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

def gradient_relu(x):
    temp = x
    temp[temp<=0] = 0
    temp[temp>0] = 1
    return temp

def Forwardpass2HiddenLayer(x, Wh1, bh1, Wh2, bh2, Wo, bo):
    zh1 = np.dot(x, Wh1.T) + bh1
    a = sigmoid(zh1) # output of layer 1
    
    zh2 = np.dot(a, Wh2.T) + bh2 
    b = sigmoid(zh2) # output of layer 2
    
    z = np.dot(b, Wo.T) + bo
    o = softmax(z)  # output 
    return o

# calculate the matching score
def AccTest(label,prediction):
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy

learningRate = 0.2
Epoch = 100
NumTrainSamples = 60000
NumTestSamples = 10000
NumInputs = 784
NumHiddenUnits = 512
NumClasses = 10

#inital weights

#hidden layer 1 = 784 x 512
Wh1 = np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumInputs)))
bh1 = np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh1 = np.zeros((NumHiddenUnits,NumInputs))
dbh1 = np.zeros((1,NumHiddenUnits))

print(np.shape(Wh1))

#hidden layer 2 = 512 x 512
#np.random.uniform(a,b,(sizeX,sizeY)) to random number from a to b with the size XY
Wh2 = np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,NumHiddenUnits)))
bh2 = np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh2 = np.zeros((NumHiddenUnits,NumHiddenUnits))
dbh2 = np.zeros((1,NumHiddenUnits))

print(np.shape(Wh2))

#Output layer = 512 x 10
Wo = np.random.uniform(-0.5,0.5,(NumClasses,NumHiddenUnits))
bo = np.random.uniform(0,0.5,(1,NumClasses))
dWo = np.zeros((NumClasses,NumHiddenUnits))
dbo = np.zeros((1,NumClasses))

print(np.shape(Wo))

#Adam
mWh1 = np.zeros((NumHiddenUnits, NumInputs))
vWh1 = np.zeros((NumHiddenUnits, NumInputs))

mbh1 = np.zeros((1, NumHiddenUnits))
vbh1 = np.zeros((1, NumHiddenUnits))

mWh2 = np.zeros((NumHiddenUnits, NumHiddenUnits))
vWh2 = np.zeros((NumHiddenUnits, NumHiddenUnits))

mbh2 = np.zeros((1, NumHiddenUnits))
vbh2 = np.zeros((1, NumHiddenUnits))

mWo = np.zeros((NumClasses, NumHiddenUnits))
vWo = np.zeros((NumClasses, NumHiddenUnits))

mbo = np.zeros((1, NumClasses))
vbo = np.zeros((1, NumClasses))

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

#Training loop
loss = []
TrainAccuracy = []
TestAccuracy = []
batch_size = 200
stochastic_samples = np.arange(NumTrainSamples)

for ep in range(Epoch):
    np.random.shuffle(stochastic_samples)
    for ite in range(0, NumTrainSamples, batch_size):
        #Forward Pass
        batch_samples = stochastic_samples[ite:ite + batch_size]
        x = x_train[batch_samples, :]
        y = y_train[batch_samples, :]
        
        zh1 = np.dot(x,Wh1.T) + bh1
        a = sigmoid(zh1)
        
        zh2 = np.dot(a, Wh2.T) + bh2
        b = sigmoid (zh2)
        
        z = np.dot(b,Wo.T) + bo
        o = softmax(z)
        
        #Calculate loss
        loss.append(-np.sum(np.multiply(y, np.log10(o))))
        #Backpropagation
        dL_dz = o - y
        
        #Back propagate error
        dh2 = np.dot(dL_dz,Wo)
        dhs2 = np.multiply(np.multiply(dh2,b),(1-b))

        dh1 = np.dot(dh2,Wh2)
        dhs1 =  np.multiply(np.multiply(dh1,a),(1-a))
        
        #Calculate the gradient
        dL_dWo = np.matmul(np.transpose(dL_dz),b)
        dL_dbo = np.mean(dL_dz) 

        dL_dWh2 = np.matmul(np.transpose(dhs2),a)
        dL_dbh2 = np.mean(dhs1)
        
        dL_dWh1 = np.matmul(np.transpose(dhs1),x)
        dL_dbh1 = np.mean(dhs2) 
        
        #Update Weights with Adam Optimizer
        mWh1 = beta1*mWh1 + (1-beta1)*dL_dWh1
        vWh1 = beta2*vWh1 + (1-beta2)*np.square(dL_dWh1)
        mWh1_hat = mWh1 / (1 - beta1)
        vWh1_hat = vWh1 / (1 - beta2)
        
        mbh1 = beta1*mbh1 + (1-beta1)*dL_dbh1
        vbh1 = beta2*vbh1 + (1-beta2)*np.square(dL_dbh1)
        mbh1_hat = mbh1 / (1 - beta1 )
        vbh1_hat = vbh1 / (1 - beta2 )
        
        mWh2 = beta1*mWh2 + (1-beta1)*dL_dWh2
        vWh2 = beta2*vWh2 + (1-beta2)*np.square(dL_dWh2)
        mWh2_hat = mWh2 / (1 - beta1)
        vWh2_hat = vWh2 / (1 - beta2)
        
        mbh2 = beta1*mbh2 + (1-beta1)*dL_dbh2
        vbh2 = beta2*vbh2 + (1-beta2)*np.square(dL_dbh2)
        mbh2_hat = mbh2 / (1 - beta1)
        vbh2_hat = vbh2 / (1 - beta2)
                
        mWo = beta1*mWo + (1-beta1)*dL_dWo
        vWo = beta2*vWo + (1-beta2)*np.square(dL_dWo)
        mWo_hat = mWo / (1 - beta1)
        vWo_hat = vWo / (1 - beta2)
        
        mbo = beta1*mbo + (1-beta1)*dL_dbo
        vbo = beta2*vbo + (1-beta2)*np.square(dL_dbo)
        mbo_hat = mbo / (1 - beta1)
        vbo_hat = vbo / (1 - beta2)
        
        #Update weights and biases
        Wh1 = Wh1 - learningRate*mWh1_hat/((np.sqrt(vWh1_hat)+epsilon)*batch_size)
        bh1 = bh1 - learningRate*mbh1_hat/((np.sqrt(vbh1_hat)+epsilon)*batch_size)
        
        Wh2 = Wh2 - learningRate*mWh2_hat/((np.sqrt(vWh2_hat)+epsilon)*batch_size)
        bh2 = bh2 - learningRate*mbh2_hat/((np.sqrt(vbh2_hat)+epsilon)*batch_size)
        
        Wo = Wo - learningRate*mWo_hat/((np.sqrt(vWo_hat)+epsilon)*batch_size)
        bo = bo - learningRate*mbo_hat/((np.sqrt(vbo_hat)+epsilon)*batch_size)
        
    #Test accuracy with random innitial weights
    prediction = Forwardpass2HiddenLayer(x_test,Wh1,bh1,Wh2,bh2,Wo,bo)
    TrainAccuracy.append(AccTest(y_test,prediction))
    clear_output(wait=True)
    print('Epoch:', ep)
    print('Accuracy:',AccTest(y_test,prediction) )
    # plt.plot([i for i, _ in enumerate(Acc)],Acc,'o')
    # plt.show()c

    prediction = Forwardpass2HiddenLayer(x_test,Wh1,bh1,Wh2,bh2,Wo,bo)
    Rate = AccTest(y_test,prediction)
    print(Rate)
    
    # #Accuracy on training data
    # train_output = Forwardpass2HiddenLayer(x_train, Wh1, bh1, Wh2, bh2, Wo, bo)
    # train_accuracy = AccTest(y_train, train_output)
    # TrainAccuracy.append(train_accuracy)

    # #Accuracy on test data
    # test_output = Forwardpass2HiddenLayer(x_test, Wh1, bh1, Wh2, bh2, Wo, bo)
    # test_accuracy = AccTest(y_test, test_output)
    # TestAccuracy.append(test_accuracy)

    # #Print accuracy and loss
    # clear_output(wait=True)
    # print('Epoch:', ep+1, ', Loss:', loss[-1], ', Train Accuracy:', train_accuracy, ', Test Accuracy:', test_accuracy)
        