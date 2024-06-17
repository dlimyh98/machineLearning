import matplotlib.pyplot as plt
import torch

class TestCaseError(Exception):
    pass

def test_allclose(input, other):
    if input.shape != other.shape:
        raise TestCaseError("Wrong dimension")
    if not torch.allclose(input, other, atol=0.0002):
        raise TestCaseError("Wrong solution")
    return True


############################################# Linear Data Generation #############################################################
def generate_1d_linear_data(weights_true, num_samples):
    uniform = torch.distributions.uniform.Uniform(0, 5)
    normal = torch.distributions.normal.Normal(0, 1)
    inputs, _ = torch.sort(uniform.sample([num_samples, 1]))
    augmented_inputs = torch.hstack((torch.ones(num_samples, 1), inputs))
    targets = augmented_inputs @ weights_true + normal.sample([num_samples])
    return inputs, targets

test_seed = 0
torch.manual_seed(test_seed)
weights_true = torch.tensor([2., 3.])
inputs, targets = generate_1d_linear_data(weights_true, 30)
fig, ax = plt.subplots()
ax.scatter(inputs, targets)
#fig.show()


############################################# PROBLEM 1 (augmenting the input matrix) #############################################################
def augment_linear_inputs(inputs):
    #print(inputs)
    padding = torch.nn.ConstantPad1d(padding=(1,0), value=1)
    augmented_inputs = padding(inputs)
    #print(augmented_inputs)
    return augmented_inputs

#augment_linear_inputs(torch.tensor([[3,3], [4,4], [5,5]]))


########################################### PROBLEM 2 (calculating gradient of error function) ######################################################
def error_gradient(weights_current, inputs, targets):
    numRows = inputs.size(dim=0)
    numCols = weights_current.size(dim=0)

    weightsReshaped = torch.reshape(weights_current, (-1,)).float()
    inputsReshaped = torch.reshape(inputs, (numRows,-1)).float()
    targetsReshaped = torch.reshape(targets, (-1,)).float()

    #print("weightsReshaped is\n", weightsReshaped)
    #print(weightsReshaped.size())
    #print("\ninputsReshaped is\n", inputsReshaped)
    #print(inputsReshaped.size())
    #print("\ntargetsReshaped is\n", targetsReshaped)
    #print(targetsReshaped.size())

    inputsTransposed = torch.transpose(input=inputsReshaped, dim0=0, dim1=1)
    firstTerm = torch.negative(torch.matmul(inputsTransposed, targetsReshaped))  ## -X^T @ Y
    secondTerm = torch.matmul(torch.matmul(inputsTransposed, inputsReshaped), weightsReshaped)  ## (X^T @ X) @ W
    finalTerm = torch.add(firstTerm, secondTerm)
    finalTermReshaped = torch.reshape(finalTerm, (-1,))

    #print("\nGradient is\n", finalTermReshaped)
    #print(finalTermReshaped.size())
    return finalTermReshaped

#error_gradient(torch.tensor([4,4,4]), torch.tensor([[1,2,2], [1,3,3]]), torch.tensor([5,5]))


################################################ PROBLEM 3 (implementing gradient descent) ###########################################################
def gradient_descent(weights_initial, inputs, targets, learning_rate, num_iterations):
    for counter in range (num_iterations):
        weights_initial = weights_initial - (learning_rate * error_gradient(weights_initial, inputs, targets))

    #print("Weights learned is \n", weights_initial)
    return weights_initial

#gradient_descent(torch.tensor([4,4,4]), torch.tensor([[1,2,2], [1,3,3]]), torch.tensor([5,5]), 0.001, 20)


####################################### PROBLEM 4 (visualize linear model, only works if dimension = 1) ##################################################
def wLinear(learnedWeights, nonAugmentedInputs):
    # given learnedWeights and input, gives PREDICTED OUTPUT, not TRUE OUTPUT
    # learnedWeights is 1x2 vector
    # nonAugmentedInputs is Nx2 vector (2 features, of which 1 is dummy feature)
    return learnedWeights[0] + learnedWeights[1] * nonAugmentedInputs    # returns Nx1 vector


# plotting w0 + w1x
def visualize_linear_model(weights_learned, inputs, targets):
    nonAugmentedInputs = (inputs[:, 1:])    # remove the augmented column of 1s

    fig, ax = plt.subplots()
    ax.set_title('Visualizing Linear Model')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.scatter(nonAugmentedInputs,targets, color = 'red')
    ax.plot(nonAugmentedInputs, wLinear(weights_learned, nonAugmentedInputs), color='green')    # x-axis should be nonAugmentedInputs, since original data does not have '1' augmented to it
    return fig

'''
torch.manual_seed(test_seed)
weights_true = torch.tensor([2., 3.])
inputs, targets = generate_1d_linear_data(weights_true, 30)
augmented_inputs = augment_linear_inputs(inputs)
weights_initial = torch.tensor([1., 1.])
learning_rate = 0.001
num_iterations = 20

weights_learned = gradient_descent(weights_initial, augmented_inputs, targets, learning_rate, num_iterations)
print(f"Model weights are {weights_learned.tolist()}")
fig = visualize_linear_model(weights_learned, augmented_inputs, targets)
fig.show()
'''


############################################### PROBLEM 5 (exploring learning rates) ##############################################################
def fitHypothesisLinear(weightsLearned, nonAugmentedInputs, numFeatures):
    numRows = nonAugmentedInputs.size(dim=0)
    numCols = nonAugmentedInputs.size(dim=1)

    predictedOutput = torch.empty(size=(numRows, numCols))

    # each weight must be matched to it's corresponding feature
    for i in range (numFeatures):
        #print(weightsLearned[i])
        #print(nonAugmentedInputs[:,i])
        #print(weightsLearned[i] * nonAugmentedInputs[:,i])
        predictedOutput[:, i] = weightsLearned[i] * nonAugmentedInputs[:,i]

    # Sum up along all columns, to give a Nx1 column vector
    # each row entry denotes the PREDICTED OUTPUT given some INPUT
    predictedOutput = predictedOutput.sum(axis=1, keepdim=True)
    return predictedOutput


def getError(trueOutputs, predictedOutputs):
    #print("trueOutputs is", trueOutputs)
    #print("\npredictedOutputs is", predictedOutputs)
    #print(torch.reshape(predictedOutputs, (-1,)))

    subError = torch.sub(trueOutputs, torch.reshape(predictedOutputs, (-1,)))
    #print("\nsubTensor is", subError)

    squaredSub = torch.square(subError)
    return (0.5 * squaredSub.sum())       # include 1/2 term as per loss function


def gradient_descent_with_logger(weights_initial, inputs, targets, learning_rate, num_iterations):
    #print("weights_initial is", weights_initial)
    #print("\ninputs is", inputs )
    #print("\ntargets is", targets)

    # add trainingError before ANY training is done
    trainingError = [getError(targets, fitHypothesisLinear(weights_initial, inputs, inputs.size(dim=1)))]

    # begin training
    for counter in range (num_iterations):
        weights_initial = weights_initial - (learning_rate * error_gradient(weights_initial, inputs, targets))
        #print('\nweightsLearned for {}th iteration is'.format(counter), weights_initial)

        predictedOutputs = fitHypothesisLinear(weights_initial, inputs, inputs.size(dim=1))
        #print("\npredictedOutputs is", predictedOutputs, "\n")
        trainingError += [getError(targets, predictedOutputs)]


    #print("\n trainingError is", trainingError)
    weights_learned = weights_initial
    return weights_learned, torch.Tensor(trainingError)

'''
weights_initial = torch.tensor([2., 3., 4., 5., 6. ])
augmented_inputs = torch.tensor([[1., 11., 12., 13., 14.],
                       [1., 15., 16., 17., 18.],
                       [1., 19., 20., 21., 22.]
                       ])
targets = torch.tensor([190,200,210])
learning_rate = 0.001
num_iterations = 5


weights_true = torch.tensor([2., 3.])
inputs, targets = generate_1d_linear_data(weights_true, 30)
augmented_inputs = augment_linear_inputs(inputs)
weights_initial = torch.tensor([1., 1.])
learning_rate = 0.001
num_iterations = 5

_, error = gradient_descent_with_logger(weights_initial, augmented_inputs, targets, learning_rate, num_iterations)
#expected_error = torch.tensor([597.8280, 349.5298, 206.3568, 123.8006, 76.1970, 48.7477])
#is_correct = test_allclose(error, expected_error)
#print(f"Public testcase for gradient_descent_with_logger: {is_correct}")
'''


############################################### PROBLEM 6 (visualising learning rates) ##############################################################
def convertToList (numIterations):
   iterationList = []
   
   for i in range(numIterations + 1):
       iterationList.append(i)

   return (iterationList)


def visualize_errors(weights_initial, inputs, targets, learning_rates, num_iterations):
    fig, ax = plt.subplots()
    ax.set_title('Visualizing Learning Rates')
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Error")

    errorLearningRates = []
    for learningRate in learning_rates:
        _, errorLearningRate = gradient_descent_with_logger(weights_initial, inputs, targets, learningRate, num_iterations)    # returns vector, corresponding to error rates as iterations increases FOR SOME learning rate
        errorLearningRates.append(errorLearningRate)

    for loopCounter, errorLearningRate in enumerate(errorLearningRates):
        convertedList = errorLearningRate.tolist()
        ax.plot(convertToList(num_iterations), convertedList, label = f'Learning Rate = {learning_rates[loopCounter]}')

    ax.legend()
    return fig

'''
torch.manual_seed(test_seed)
weights_true = torch.tensor([2., 3.])
inputs, targets = generate_1d_linear_data(weights_true, 30)
augmented_inputs = augment_linear_inputs(inputs)
weights_initial = torch.tensor([1., 1.])
learning_rates = [ 0.0003, 0.001, 0.003, 0.009 ]
num_iterations = 20

fig = visualize_errors(weights_initial, augmented_inputs, targets, learning_rates, num_iterations)
fig.axes[0].set_ylim([0, 1000])
fig.show()
'''


############################################# Non-Linear Data Generation #############################################################
def generate_1d_nonlinear_data(weights_true, num_samples):
    uniform = torch.distributions.uniform.Uniform(1, 5)
    normal = torch.distributions.normal.Normal(0, 0.1)
    inputs, _ = torch.sort(uniform.sample([num_samples, 1]), dim=0)
    targets = weights_true[0] * torch.ones([num_samples]) + \
              weights_true[1] * inputs[:,0] + \
              torch.log(inputs[:,0] ** weights_true[2]) + \
              normal.sample([num_samples])
    return inputs, targets

torch.manual_seed(test_seed)
weights_true = torch.Tensor([3., -2., 5.])
inputs, targets = generate_1d_nonlinear_data(weights_true, 50)

fig, ax = plt.subplots()
ax.scatter(inputs, targets)
#fig.show()


############################################# PROBLEM 7 (implement basis function) #############################################################
def transformBasis(dataPoint):
    return [1, dataPoint.item(), torch.log(dataPoint).item()]

def basis_function(inputs):
    numRows = inputs.size(dim=0)
    mappedInputs = torch.empty(size=(numRows, 3))

    for i in range(numRows):
        mappedInputs[i, :] = torch.FloatTensor(transformBasis(inputs[i,:]))

    return mappedInputs

'''
torch.manual_seed(test_seed)
weights_true = torch.Tensor([3., -2., 5.])
inputs, targets = generate_1d_nonlinear_data(weights_true, 3)

mapped_inputs = basis_function(inputs.clone())
expected_mapped_inputs = torch.tensor([[1.0000, 1.3539, 0.3030], [1.0000, 2.9850, 1.0936], [1.0000, 4.0729, 1.4044]])
is_correct = test_allclose(mapped_inputs, expected_mapped_inputs)
print(f"Public testcase for basis_function: {is_correct}")
'''


############################################# PROBLEM 8 (visualise non-linear model) #############################################################
def reverseBasis(basisData):
    return[basisData[1]]

def wNonLinear(weights_learned, basis_inputs):
    return weights_learned[0] * basis_inputs[:,0] + weights_learned[1] * basis_inputs[:,1] + weights_learned[2] * basis_inputs[:,2]

def plot_error_curve(errors, learning_rate):
    fig, ax = plt.subplots()
    ax.set_title('Errors of Non-Linear Model')
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("$Error$")
    ax.plot(errors, label = f'Learning Rate = {learning_rate}')
    ax.legend()
    ax.set_ylim(0, 30)
    return fig

def visualize_nonlinear_model(weights_learned, basis_inputs, targets):
    # get the original inputs
    numRows = basis_inputs.size(dim=0)
    originalInputs = torch.empty(size=(numRows, 1))
    
    for i in range (numRows):
        originalInputs[i,:] = torch.FloatTensor(reverseBasis(basis_inputs[i,:]))

    fig, ax = plt.subplots()
    ax.set_title('Visualizing Non-Linear Model')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$t$")
    ax.scatter(originalInputs, targets, color = 'red')
    ax.plot(originalInputs, wNonLinear(weights_learned, basis_inputs), color='green')

    return fig

'''
torch.manual_seed(test_seed)
weights_true = torch.Tensor([3., -2., 5.])
inputs, targets = generate_1d_nonlinear_data(weights_true, 50)
mapped_inputs = basis_function(inputs)
weights_initial = torch.Tensor([1., 1., 1.])
learning_rate = 0.001
num_iterations = 20

weights_learned, errors = gradient_descent_with_logger(weights_initial, mapped_inputs, targets, learning_rate, num_iterations)
print(f"Model has an error of {errors[-1]:.3f}")
fig = visualize_nonlinear_model(weights_learned, mapped_inputs, targets)
fig.show()

fig = plot_error_curve(errors, learning_rate)
fig.show()
'''


############################################# PROBLEM 9 (adjust hyperparameters, so error < 0.6) #############################################################
def fit_nonlinear_model(inputs, targets):
    mapped_inputs = basis_function(inputs)
    weights_initial = torch.Tensor([1.0, 1.0, 1.0])
    ### start modifying the code here ###
    learning_rate = 0.0030
    num_iterations = 1450
    ### stop modifying the code here ###
    weights_learned, errors = gradient_descent_with_logger(weights_initial, mapped_inputs, targets, learning_rate, num_iterations)
    fitted_figure = visualize_nonlinear_model(weights_learned, mapped_inputs, targets)
    return weights_learned, errors, fitted_figure

'''
torch.manual_seed(test_seed)
weights_learned, errors, fitted_figure = fit_nonlinear_model(inputs, targets)
error_relative_string = "more" if errors[-1] > 0.6 else "less"
print(f"Model has an error of {errors[-1]:.3f}, which is {error_relative_string} than the required error of 0.6.")
fitted_figure.show()

error_figure = plot_error_curve(errors, learning_rate)
error_figure.show()
'''
