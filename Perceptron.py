import numpy as np
import matplotlib.pyplot as plt

#prepare the data
#@files - input file to read
#@indices - tuple of indices to fetech from the file
#@labls - dictionary of labels and their corresponding category value

def prepare_data(files,indices,labls):
  with open(files) as file:
    data = file.read()

  #data = data.replace('class-','')
  data = data.splitlines()
  data = [x.split(',') for x in data]
  data = np.array(data)
  labels = data[indices[0]:indices[1],4:5]
  data = data[indices[0]:indices[1],0:4]

  data = np.array(data, dtype="float64")

  for key,value in labls.items():

    for i in range(len(labels)):
      if(labels[i] == key):
        labels[i] = value

  return (data,np.array(labels, dtype="float64"))

# Class Perceptron
#@@ __init__ - params : {input_size,gamma}
####@ input_size - input size
####@ gamma - Coeficient of the l2 regularization default is 0(no regularisation)

#@@ predict - params : {inputs}
#@@ activation - params : {input} returns threshold value
#@@ train - params : {inputs,labels,iterations}
####@ inputs - list of inputs
####@ labels - list of labels
####@ iterations - Iteration count

#@@ data_matrix - params : {inputs,labels} - Prints the data matrix for the input list

class Perceptron:

  def __init__(self,inp_size, gamma = 0):
    self.weights = np.zeros(inp_size)
    self.gamma = gamma
    self.bias = 0 #np.zeros((2,1))

  def predict(self,inputs):
    #return np.dot(input,self.weights)
    return self.weights.T.dot(inputs) + self.bias


  def activation(self,input):
    if input <= 0.0:
      return -1.0
    else:
      return 1.0

  def train(self, inputs,labels,iterations):
    self.inputs = inputs
    self.labels = labels

    for i in range(iterations):
      for index, x in enumerate(self.inputs):
        ya = self.labels[index] * self.predict(x)

        if ya[0] <= 0:
          self.weights = self.weights +  self.labels[index]*x + 1/2 *self.gamma * (np.sum(self.weights)**2)
          self.bias = self.labels[index] + self.bias

  def data_matrix(self, inputs, labels):

    matrix = np.zeros((2,2))

    for i in range(len(inputs)):
      p = self.activation(self.predict(inputs[i]))
      #print(">>>", p, ">>>", labels[i])
      if(p == 1):
        if(labels[i] == 1):
          matrix[0,0] += 1
        else:
          matrix[0,1] += 1


      elif(p == -1):
        if(labels[i] == -1):
          matrix[1,1] += 1
        else:
          matrix[1,0] += 1
    
    #print("confusion Matrix : ", matrix)
    accuracy = (matrix[0,0] + matrix[1,1]) / (matrix[0,0] + matrix[1,0] + matrix[0,1] + matrix[1,1]) if(matrix[0,0] + matrix[1,0] + matrix[0,1] + matrix[1,1] > 0) else 0
    precision = matrix[0,0] / (matrix[0,0] + matrix[0,1]) if(matrix[0,0] + matrix[0,1] > 0) else 0
    recall = matrix[0,0] / (matrix[0,0] + matrix[1,0]) if(matrix[0,0] + matrix[1,0] > 0) else 0
    f_score =  (2 * precision * recall) / (precision + recall) if(precision + recall > 0) else 0

    print("Precision : %s" %precision)
    print("Accuracy : %s" %accuracy)
    print("Recall : %s" %recall)
    print("F-Score : %s" %f_score)

    return (accuracy,precision,recall,f_score)