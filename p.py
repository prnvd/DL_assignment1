
import numpy as np 
from sklearn.model_selection import train_test_split


class perceptron:

    def __init__(self,learning_rate,epochs):

        self.lr = learning_rate
        self.epochs = epochs
        self.activation_func= self.activation_func
        self.weight = None
        self.bias= None
    
    #set activation function
    def activation_func(self,x):
        return np.where(x>=0,1,0)       


    def prep_train(self,X,y):
        
        #initializing the weights(size:features) and bias (experimental)
        self.weight = np.array([1,0.5,1,0.5,1,0.5,1,0.5])
        self.bias = 1


        for i in range(self.epochs):

            for x_id,x_val in enumerate(X):

                #calculating partial sum
                partial_sum = np.dot(x_val, self.weight) + self.bias
                #activating partial sum
                y_pred = self.activation_func(partial_sum)

                #updating the weights using learning rate
                update = self.lr * (y[x_id] - y_pred)

                self.weight += update * x_val
                self.bias += update


    def predict(self,X):
        partial_sum = np.dot(X,self.weight)+self.bias
        predicted_val = self.activation_func(partial_sum)

        return predicted_val


 




#taking data from csv files
X = np.genfromtxt('data/diabetes_X.csv',delimiter=',')
y = np.genfromtxt('data/diabetes_y.csv',delimiter=',')

#normalizing the data 
X = (X - np.min(X))/(np.max(X)-np.min(X))

#spliting the data into training and testing datas
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)


#initializing the perceptron instance
p = perceptron(learning_rate=0.018, epochs=1000)
p.prep_train(X_train, y_train)
predictions = p.predict(X_test)

#function to calculate accuracy
def accuracy(y_pred, y):
    return (np.sum(y == y_pred)/len(y))


print("Perceptron accuracy: ", accuracy(y_test, predictions)*100,"%")
print("Weights for this test: ",p.weight)




