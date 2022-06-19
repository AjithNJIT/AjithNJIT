#!/usr/bin/env python
# coding: utf-8

# In[107]:


from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt


# In[218]:


## 1.(a) X with 20 samples with value of lamda = 1/mu = 0.25
#x=np.array([0,1,2,3,4,5])
x = np.arange(0, 20, 0.5)
expon.pdf(x, loc=0, scale=4)
#fig,ax = plt.subplots(figsize=(16,6))
#ax.plot(x,expon.pdf(x,loc=0, scale=4))
#plt.show


# In[220]:


x = np.arange(0, 20, 0.5)
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,expon.pdf(x,loc=0, scale=1))
plt.show


# In[219]:


x = np.arange(0.1, 20, 0.5)
y = -np.log(x)

fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,y)
plt.ylabel('-log(x)')
plt.xlabel('x')
plt.title('Negative log-likelihood range')
plt.show()


# In[221]:


## 1.(b) X with 200 samples with value of lamda = 1/mu = 0.25
x = np.arange(0, 200, 2.5)
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,expon.pdf(x,loc=0, scale=1))
plt.show


# In[222]:


#Part 2

#a. Plotting the Negatvive Likelyhood of the Exponential function

x = np.arange(0, 20, 0.5)
fig,ax = plt.subplots(figsize=(6,4))
ax.plot(x,expon.logpdf(x,loc=0, scale=1))
plt.show


# In[189]:


##Mean = 1/lambda
##variance = 1/lambda square

#Mean = 4
#(1/4)
#lam = 0.25
#Variance = 16


# In[223]:


#Part 2

#a. Plotting the Negatvive Likelyhood of the Exponential function

###Calculate the Sum and total number of samples from the given data
Sum = np.sum(x)
print("Sum of the values of X is =", Sum)
n = len(x)
print("Length of the values of X is =", n)
##print("lambda =" , Sum/n)
lamb = np.arange(0.2, 20, 0.5)
L = n * np.log(lamb) - lamb * Sum
fig,ax = plt.subplots(figsize=(6,4))
#plt.scatter(x, L)
ax.plot(x,L)
plt.show


# In[198]:


##Part 2
##B.What is the lambda MLE of the generated data? (10 points)

TRUE_LAMBDA = 0.25
X = np.random.exponential(TRUE_LAMBDA, 200)


# In[199]:


def exp_lamda_MLE(X):
    T = len(X)
    s = sum(X)
    return s/T


# In[200]:


print("lambda estimate:", str(exp_lamda_MLE(X)))


# In[194]:


# The scipy version of the exponential distribution has a location parameter
# that can skew the distribution. We ignore this by fixing the location
# parameter to 0 with floc=0
_, l = scipy.stats.expon.fit(X, floc=0)


# In[215]:


###2.C Plot the estimated lambda vs iterations to showcase convergence towards the true lambda (10 points)

pdf = scipy.stats.expon.pdf
x = range(0, 2)
plt.hist(X, bins=x, density='true')
plt.plot(pdf(x, scale=0.25))
plt.xlabel('Value')
plt.ylabel('Observed Frequency')
plt.legend(['Fitted Distribution PDF', 'Observed Data', ]);


# In[ ]:


##Part 3.A: Write the expression of the Negative Log Likelihood function ATTACHED (Word Document)


# In[ ]:


##Part 3.B: Write the parameters  and the  that minimize the NLL (10 points) ATTACHED(Word Document)


# In[10]:


##Part 3.C
##Write a Python script that uses SGD to converge to  and for the following dataset (20 points)

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

def mean_squared_error(y_true, y_predicted):
	
	# Calculating the loss or cost
	cost = np.sum((y_true-y_predicted)**2) / len(y_true)
	return cost

# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned
def gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001,
					stopping_threshold = 1e-6):
	
	# Initializing weight, bias, learning rate and iterations
	current_weight = 0.1
	current_bias = 0.01
	iterations = iterations
	learning_rate = learning_rate
	n = float(len(x))
	
	costs = []
	weights = []
	previous_cost = None
	
	# Estimation of optimal parameters
	for i in range(iterations):
		
		# Making predictions
		y_predicted = (current_weight * x) + current_bias
		
		# Calculationg the current cost
		current_cost = mean_squared_error(y, y_predicted)

		# If the change in cost is less than or equal to
		# stopping_threshold we stop the gradient descent
		if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
			break
		
		previous_cost = current_cost

		costs.append(current_cost)
		weights.append(current_weight)
		
		# Calculating the gradients
		weight_derivative = -(2/n) * sum(x * (y-y_predicted))
		bias_derivative = -(2/n) * sum(y-y_predicted)
		
		# Updating weights and bias
		current_weight = current_weight - (learning_rate * weight_derivative)
		current_bias = current_bias - (learning_rate * bias_derivative)
				
		# Printing the parameters for each 1000th iteration
		print(f"Iteration {i+1}: Cost {current_cost}, Weight 		{current_weight}, Bias {current_bias}")
	
	
	# Visualizing the weights and cost at for all iterations
	plt.figure(figsize = (8,6))
	plt.plot(weights, costs)
	plt.scatter(weights, costs, marker='o', color='red')
	plt.title("Cost vs Weights")
	plt.ylabel("Cost")
	plt.xlabel("Weight")
	plt.show()
	
	return current_weight, current_bias


def main():
	
	# Data
	X = np.array([8, 16, 22, 33, 50, 51])
	Y = np.array([5, 20, 14, 32, 42, 58])

	# Estimating weight and bias using gradient descent
	estimated_weight, eatimated_bias = gradient_descent(X, Y, iterations=2000)
	print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {eatimated_bias}")

	# Making predictions using estimated parameters
	Y_pred = estimated_weight*X + eatimated_bias

	# Plotting the regression line
	plt.figure(figsize = (8,6))
	plt.scatter(X, Y, marker='o', color='red')
	plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
			markersize=10,linestyle='dashed')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()

	
if __name__=="__main__":
	main()


# In[ ]:




