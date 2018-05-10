# CS231n Assignment 2017 Spring

## Assignment 1
### KNN
**Compute the distance between each test point in X and each training point**
we can use loops or matrix calculation to finish this task
- two loops
```
    for i in range(num_test):
      for j in range(num_train):
        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
```
- one loop
```
    for i in range(num_test):
      dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i]), axis = 1))
```
- no loop
```
    aSumSquare = np.sum(np.square(X),axis=1)
    bSumSquare = np.sum(np.square(self.X_train),axis=1)
    mul = np.dot(X, self.X_train.T)
    dists = np.sqrt(aSumSquare[:,np.newaxis]+bSumSquare-2*mul)
```

**predict labels**
```
    for i in range(num_test):
      closest_y = []

      #find the k nearest neighbors of the ith testing point
      sort = np.argsort(dists[i])

      for index in range(k):
            closest_y.append(self.y_train[sort[k]])

      #find the most common label in the list closest_y of labels
      y_pred[i] = np.bincount(closest_y).argmax()
```

### SVM
**SVM loss function, vectorized implementation**
```
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[range(num_train), list(y)].reshape(-1,1)
  # the correct one does not count
  margins = np.maximum(0, scores - correct_class_scores +1)
  margins[range(num_train), list(y)] = 0
  loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)
```
**SVM gradient, vectorized implementation**
```
  coeff_mat = np.zeros((num_train, num_classes))
  coeff_mat[margins > 0] = 1
  coeff_mat[range(num_train), list(y)] = 0
  coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)

  dW = (X.T).dot(coeff_mat)
  dW = dW/num_train + reg*W
```

### Softmax
softmax loss function and gradient, use loops to finish explicitly
```
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
     scores = X[i].dot(W)
     shift_scores = scores - max(scores)
     loss_i = - shift_scores[y[i]] + np.log(sum(np.exp(shift_scores)))
     loss += loss_i
     for j in range(num_classes):
         softmax_output = np.exp(shift_scores[j])/sum(np.exp(shift_scores))
         if j == y[i]:
             dW[:,j] += (-1 + softmax_output) * X[i]
         else:
             dW[:,j] += softmax_output *X[i]

  loss /= num_train
  loss +=  0.5 * reg * np.sum(W * W)
  dW = dW / num_train + reg * W
 ```
 softmax loss function and gradient vectorized
```
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_output[range(num_train), list(y)]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dS = softmax_output.copy()
  dS[range(num_train), list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW / num_train + reg * W
```
### 2 Layers Neural Network
Forward pass
```
  h = np.maximum(0, X.dot(W1) + b1)
  scores = h.dot(W2) + b2
```

Compute the loss with softmax classifier
```
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  loss = -np.sum(np.log(softmax_output[range(N), list(y)]))
  loss /= N
  loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
```

Backward pass and gradients
```
  dscores = softmax_output.copy()
  dscores[range(N), list(y)] -= 1
  dscores /= N
  grads['W2'] = h.T.dot(dscores) + reg * W2
  grads['b2'] = np.sum(dscores, axis=0)

  dh = dscores.dot(W2.T)
  dh_ReLu = (h > 0) * dh
  grads['W1'] = X.T.dot(dh_ReLu) + reg * W1
  grads['b1'] = np.sum(dh_ReLu, axis=0)
```

## Assignment 2
### Fully Connected Network

to be continued...