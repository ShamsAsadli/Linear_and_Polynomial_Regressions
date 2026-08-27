from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#  Load dataset 
credit_approval = fetch_ucirepo(id=27)
X = credit_approval.data.features.copy()
y = credit_approval.data.targets

# Encode categorical (object) columns in X 
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = X[col].astype(str)         # handle NaN-as-float edge case (undefined or unrepresentable values)
    X[col] = le.fit_transform(X[col])

# Encode target
y = y.copy()
for col in y.select_dtypes(include='object').columns:
    y[col] = le.fit_transform(y[col].astype(str))

#  Handle missing values 
X.fillna(X.median(numeric_only=True), inplace=True)

#  Convert to numpy 
X = X.values.astype(float)
y = y.values.ravel().astype(float)    

#  Scale features 
scaler = StandardScaler()
X = scaler.fit_transform(X)

#  Add bias column 
X = np.hstack([np.ones((X.shape[0], 1)), X])

# Initialize weights 
w = np.zeros(X.shape[1])

#  Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Linear combination 
def z(x, w):
    return x @ w

#  Sigmoid function 
def sigmoid(z_val):
    return np.where(
        z_val >= 0,
        1 / (1 + np.exp(-z_val)),
        np.exp(z_val) / (1 + np.exp(z_val))
    )

#  Cross-Entropy Loss  L = -(1/N)*sum(y*log(p) + (1-y)*log(1-p)) 
def ce_Loss(x, y, w):
    N   = x.shape[0]
    p   = sigmoid(z(x, w))
    eps = 1e-15                         
    p   = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

print("Initial CE Loss:", ce_Loss(X_train, y_train, w))

#  Gradient of Cross-Entropy Loss  ∂L/∂w = (1/N) * Xᵀ(p - y) 
def grad_ceL(x, y, w):
    N = x.shape[0]
    p = sigmoid(z(x, w))
    return x.T @ (p - y) / N

print("Initial gradient:", grad_ceL(X_train, y_train, w))

#  Gradient Descent 
alpha      = 0.1
iterations = 5000

print("Initial CE Loss:", ce_Loss(X_train, y_train, w))

for i in range(iterations):
    grad = grad_ceL(X_train, y_train, w)
    w    = w - alpha * grad
    loss = ce_Loss(X_train, y_train, w)
    if (i + 1) % 100 == 0:
        print(f"Iteration {i+1:4d} | Loss: {loss:.6f}")

print("\nOptimal weights:", w)
print("Minimum CE Loss:", ce_Loss(X_train, y_train, w))

#  Predict on test set 
y_pred_prob = sigmoid(z(X_test, w))
y_pred = np.where(y_pred_prob <= 0.5, 0, 1)  

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["-  (Rejected)", "+  (Approved)"]))
print("Accuracy:", accuracy_score(y_test, y_pred))


#                        OPEN-ENDED QUESTION:
#       How do you evaluate the performance of an applied machine learning model? 
#  Which evaluation metrics are most important for the correct implementation of a given task, and why?

# To evaluate a machine learning model, we measure how accurately it makes predictions on new, unseen data. 
# For this task, the model decides whether to approve (+) or reject (-) a credit application. 
# Here, both types of errors have financial consequences:
# If the model approves someone who shouldn't be approved (false positive) → the bank loses money on a bad loan.
# If the model rejects someone who should be approved (false negative) → the bank loses a good customer and potential profit.
# Since both errors matter, no single mistake is dramatically worse than the other. 
# This means we should focus on F1-score, which balances precision and recall together, giving us a fair overall picture of performance.
# We also look at:

# Accuracy — percentage of correct decisions overall (reliable here only if the dataset is balanced between approved and rejected)
# Precision — out of all applications the model approved, how many were actually good applicants
# Recall — out of all actually good applicants, how many did the model correctly approved
# F1-score — the balance between precision and recall
#  Confusion matrix — shows the exact breakdown of correct and incorrect approvals/rejections

# In summary, for a credit approval task like this one, F1-score is the most important metric because 
# we need to balance the risk of approving bad applicants against the cost of rejecting good ones. 
# Neither error should be completely ignored.