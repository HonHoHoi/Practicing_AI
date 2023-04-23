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
    temp[temp>=0] = 1100
    temp[temp<0] = 1
    return temp

def Forwardpass2HiddenLayer(x, Wh1, bh1, Wh2, bh2, Wo, bo):
    zh1 = np.dot(x, Wh1.T) + bh1
    a = sigmoid(zh1) # output of layer 1
    
    zh2 = np.dot(a, Wh2.T) + bh2 
    b = sigmoid(zh2) # output of layer 2
    
    z = np.dot(b, Wo.T) + bo
    o = softmax(z)  # output 
    return o

def AccTest(label,prediction): # calculate the matching score
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

#Momentum
Epoch = 20
loss = []
Acc = []
Batch_size = 200
learningRate = 0.9
stochastic_samples = np.arange(NumTrainSamples)

dWo_prev = dWo.copy()
dbo_prev = dbo.copy()
dbh2_prev = dbh2.copy()
dWh2_prev = dWh2.copy()
dWh1_prev = dWh1.copy()
dbh1_prev = dbh1.copy()

#Loop through training examples
for ep in range (Epoch):
    np.random.shuffle(stochastic_samples)
    for ite in range (0,NumTrainSamples,Batch_size): 
        #feed fordware propagation 
        Batch_samples = stochastic_samples[ite:ite+Batch_size]
        x = x_train[Batch_samples, :]
        y = y_train[Batch_samples, :]
        
        #Hidden layer computation
        zh1 = np.dot(x,Wh1.T) + bh1
        a = sigmoid(zh1)
        zh2 = np.dot(a, Wh2.T) + bh2
        b = sigmoid (zh2)
        z = np.dot(b,Wo.T) + bo
        o = softmax(z)
        
        #Calculate loss
        loss.append(-np.sum(np.multiply(y,np.log10(o))))
        #Calculate the output error 
        d = o - y
        
        #Backpropagationn
        #Back propagate error
        dh2 = np.dot(d,Wo)
        dhs2 = np.multiply(np.multiply(dh2,b),(1-b))

        dh1 = np.dot(dh2,Wh2)
        dhs1 =  np.multiply(np.multiply(dh1,a),(1-a))
        
        #Calculate the gradient
        dWo = np.matmul(np.transpose(d),b)
        dbo = np.mean(d) 

        dWh2 = np.matmul(np.transpose(dhs2),a)
        dbh2 = np.mean(dhs1)
        
        dWh1 = np.matmul(np.transpose(dhs1),x)
        dbh1 = np.mean(dhs2) 
        
        #Update weights and biases with momentum
        Wo = Wo - ( learningRate * dWo + dWo_prev ) / Batch_size
        bo = bo - learningRate * dbo + dbo_prev

        Wh2 = Wh2 - ( learningRate * dWh2 + dWh2_prev ) / Batch_size
        bh2 = bh2 - learningRate * dbh2 + dbh2_prev

        Wh1 = Wh1 - ( learningRate * dWh1 + dWh1_prev ) / Batch_size
        bh1 = bh1 - learningRate * dbh1 + dbh1_prev
        
        dwo_prev = 0.9 * dWo
        dbo_prev = 0.9 * dbo
        
        dbh2_prev = 0.9 * dbh2
        dWh2_prev = 0.9 * dWh2
        
        dWh1_prev = 0.9 * dWh1
        dbh1_prev = 0.9 * dbh1
# #Calculate the accuracy after each epoch
# #Feedforward on the test set
# zh1 = np.dot(x_test, Wh1.T) + bh1
# a = sigmoid(zh1)
# zh2 = np.dot(a, Wh2.T) + bh2
# b = sigmoid(zh2)
# z = np.dot(b, Wo.T) + bo
# o = softmax(z)

    #Test accuracy with random innitial weights
    prediction = Forwardpass2HiddenLayer(x_test,Wh1,bh1,Wh2,bh2,Wo,bo)
    Acc.append(AccTest(y_test,prediction))
    clear_output(wait=True)
    print('Epoch:', ep)
    print('Accuracy:',AccTest(y_test,prediction) )
    # plt.plot([i for i, _ in enumerate(Acc)],Acc,'o')
    # plt.show()c

    prediction = Forwardpass2HiddenLayer(x_test,Wh1,bh1,Wh2,bh2,Wo,bo)
    Rate = AccTest(y_test,prediction)
    print(Rate)
    