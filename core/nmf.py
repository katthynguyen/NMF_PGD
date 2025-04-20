import numpy as np

# NMF Implementation with Gradient Descent


class NMF:
    def __init__(self, n_components, max_iter=50, learning_rate=0.01, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol  # Tolerance for early stopping
        self.W = None
        self.H = None
        self.loss_history = []  # Lưu trữ lịch sử lỗi

    def _loss(self, V, W, H):
        """Compute the Frobenius norm of the reconstruction error."""
        return np.sum((V - W @ H) ** 2)

    def fit(self, V):
        """Train NMF using gradient descent."""
        n_samples, n_features = V.shape

        # Initialize W and H with small positive random values
        self.W = np.abs(np.random.randn(n_samples, self.n_components)) * 0.01 + 0.1
        self.H = np.abs(np.random.randn(self.n_components, n_features)) * 0.01 + 0.1

        # Reset loss history
        self.loss_history = []

        # Store previous loss for early stopping
        prev_loss = float("inf")
        lr = self.learning_rate

        for iteration in range(self.max_iter):
            # Compute gradients
            WH = self.W @ self.H
            error = V - WH

            # Gradient for W: dL/dW = -2 * (V - WH) @ H.T
            grad_W = -2 * error @ self.H.T
            # Gradient for H: dL/dH = -2 * W.T @ (V - WH)
            grad_H = -2 * self.W.T @ error

            # Update W and H using gradient descent
            self.W -= lr * grad_W
            self.H -= lr * grad_H

            # Ensure non-negativity by clipping
            self.W = np.maximum(self.W, 0)
            self.H = np.maximum(self.H, 0)

            # Compute loss and store it
            loss = self._loss(V, self.W, self.H)
            self.loss_history.append(loss)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {loss}")

            # Early stopping: stop if loss improvement is small
            if abs(prev_loss - loss) < self.tol:
                print(f"Early stopping at iteration {iteration}, Loss: {loss}")
                break
            prev_loss = loss

            # Adaptive learning rate: reduce lr if loss increases
            if iteration > 0 and loss > prev_loss:
                lr *= 0.5  # Reduce learning rate if loss increases

        return self.loss_history  # Trả về lịch sử lỗi

    def transform(self, V):
        """Transform input data V into the lower-dimensional space."""
        return np.dot(self.W.T, V)
