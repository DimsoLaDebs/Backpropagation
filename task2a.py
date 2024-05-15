import numpy as np
import utils
import typing

np.random.seed(1)

def find_mean_std():
    X_train, Y_train, X_val, Y_val=utils.load_full_mnist()
    X = np.vstack((X_train, X_val))
    mean = np.mean(X)
    std = np.std(X)
    return mean, std


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    mean, std = 33.63, 78.97 #founded with function find_mean_std, cf report

    # Normalize images
    X_normalized = (X - mean) / std
    
    # Apply the bias trick: add a column of ones to the end of X_normalized
    ones = np.ones((X_normalized.shape[0], 1))
    X_pre_processed = np.hstack((X_normalized, ones))
    
    return X_pre_processed


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    # Compute the cross-entropy loss
    num_examples = targets.shape[0]
    loss = np.sum(-targets * np.log(outputs)) / num_examples
    
    return loss



class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            3
        )  # Always reset random seed before weight init to get comparable results.
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if self.use_improved_weight_init:
                w = np.random.randn(prev, size) * np.sqrt(1. / prev)  # Improved weight initialization
            else:
                w = np.zeros(w_shape)  # Original initialization
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)

        self.hidden_layer_output = [X]

        #Hidden layer forward
        for i in range(len(self.neurons_per_layer) - 1):
            H = np.dot(self.hidden_layer_output[i], self.ws[i])

            if self.use_improved_sigmoid:
                yH = 1.7159 * np.tanh(2/3 * H)
            else :
              yH = 1 / (1 + np.exp(H)) #Sigmoid           
           
            self.hidden_layer_output.append(yH)

        #Output Layer Forward
        Z = np.dot(yH, self.ws[-1])
        #Softmax
        exp_Z = np.exp(Z)
        y = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        return y 
        

        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...


    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)

        

        #Grad of output layer
        deltas = [-(targets - outputs)]
        num_examples = self.hidden_layer_output[-1].shape[0]
        output_grad = np.dot(self.hidden_layer_output[-1].T, deltas[-1]) / num_examples
        self.grads[-1] = output_grad
        
        #Grad of the hidden layer
        for i in range(len(self.neurons_per_layer)-1):
            delta_j = np.dot(deltas[-1-i], self.ws[-1-i].T)
            if self.use_improved_sigmoid:
                # Calculate the derivative of the improved sigmoid function
                sigmoid_deriv = 1.7159 * (2/3 * (1 - (self.hidden_layer_output[-1-i]/1.7159)**2))
            else:
                # Existing code for standard sigmoid derivative
                sigmoid_deriv = -self.hidden_layer_output[-1-i] * (1-self.hidden_layer_output[-1-i])
            delta_j = sigmoid_deriv * delta_j
            deltas.append(delta_j)
            deltas.reverse()


            self.grads[-2-i] = np.dot(self.hidden_layer_output[-2-i].T, delta_j)/self.hidden_layer_output[-2-i].shape[0]

        '''#Grad of output layer
        delta_k = -(targets - outputs)
        num_examples = self.hidden_layer_output.shape[0]
        output_grad = np.dot(self.hidden_layer_output.T, delta_k) / num_examples
        self.grads[1] = output_grad
        
        #Grad of the hidden layer
        delta_j = np.dot(delta_k, self.ws[1].T)
        if self.use_improved_sigmoid:
            # Calculate the derivative of the improved sigmoid function
            sigmoid_deriv = 1.7159 * (2/3 * (1 - (self.hidden_layer_output/1.7159)**2))
        else:
            # Existing code for standard sigmoid derivative
            sigmoid_deriv = -self.hidden_layer_output * (1-self.hidden_layer_output)
        delta_j = sigmoid_deriv * delta_j


        self.grads[0] = np.dot(X.T, delta_j)/X.shape[0]'''



        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    num_examples = Y.shape[0]
    Y_encoded = np.zeros((num_examples, num_classes))
    for i in range(num_examples):
       label = int(Y[i])  # Extract the label
       Y_encoded[i, label] = 1
    return Y_encoded


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [128, 64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
