# Perceptron

##Steps : 

	1) Use prepare_data() method to prepare the inputs and labels. The prepare data accepts three parameters
		#@files - input file to read
		#@indices - tuple of indices to fetch from the file
		#@labls - dictionary of labels and their corresponding category value for encoding labels

		example usage : data1,label1 = prepare_data('train.data', indices = (0,80), labls = {"class-1": -1, "class-2": 1})

	2) Create an object of Perceptron class. The constructor recieves two parameters
		#@@ __init__ - params : {input_size,gamma}
		####@ input_size - input size
		####@ gamma - Coeficient of the l2 regularization default is 0 (no regularisation)


	3) Implement a binary Perceptron
    ######### Prepare the data ###########
		data1,label1 = prepare_data('train.data',(0,80), labls = {"class-1": -1, "class-2": 1}) 
		Perceptron1 = Perceptron(4) 	############ Constructor with 4 input size ################
		Perceptron1.train(data1,label1,20)	############ Train the Perceptron ##############

		print(Perceptron1.data_matrix(data1,label1)) ############ View the data matrix ##############

		print(Perceptron1.activation(Perceptron1.predict(inputs = [1,2,3,4])))  ###### Prediction of new data point #####


	4) Multiple classes
		data1,label1 = prepare_data('train.data',(0,120), labls = {"class-1": 1, "class-2": -1, "class-3": -1})
		Perceptron1 = Perceptron(4) 			  ############ Constructor with 4 input size ################
		Perceptron1.train(data1,label1,20)	############ Train the Perceptron ##############

		print(Perceptron1.data_matrix(data1,label1)) ############ View the data matrix ##############


	5) Perceptron with l2 Regularisation

		data1,label1 = prepare_data('train.data',(0,120), labls = {"class-1": 1, "class-2": -1, "class-3": -1})
		Perceptron1 = Perceptron(4, gamma = 0.01) 			############ Constructor with 4 input size ################
		Perceptron1.train(data1,label1,20)							############ Train the Perceptron ##############

		print(Perceptron1.data_matrix(data1,label1)) ############ View the data matrix ##############
