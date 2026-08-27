from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#  Load dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15)
X = breast_cancer_wisconsin_original.data.features.copy()
y = breast_cancer_wisconsin_original.data.targets

#  Examine & fix missing values / object columns 
print(X.dtypes)
print(X.isnull().sum())

# The 'Bare_nuclei' column contains missing values; fill with column median
X['Bare_nuclei'] = X['Bare_nuclei'].replace('?', np.nan).astype(float)
X.fillna(X.median(numeric_only=True), inplace=True)

# Convert object columns to numeric (none remain after above, but kept as guard)
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype(float)

#  Convert to numpy 
X = X.values                          
y = y.values.ravel()                 

#  Remap labels 2 → -1, 4 → 1 
y = np.where(y == 2, -1, 1)

# Scale features 
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias column (column of ones) 
X = np.hstack([np.ones((X.shape[0], 1)), X])   

#  Initialize weights 
w = np.zeros(X.shape[1])             

# Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear combination z = X @ w 
def z(x, w):
    return x @ w

#  Hinge Loss  L = (1/N) * sum(max(0, 1 - y * z)) 
def hinge_Loss(x, y, w):
    margins = 1 - y * z(x, w)
    return np.mean(np.maximum(0, margins))

print("Initial Hinge Loss:", hinge_Loss(X_train, y_train, w))

#  Gradient of Hinge Loss 
def grad_HL(x, y, w):
    N = x.shape[0]
    margins = 1 - y * z(x, w)          
    # Indicator: 1 where margin > 0 (misclassified / inside margin)
    mask = (margins > 0).astype(float)  
    # gradient contribution per sample: -y_i * x_i  when margin > 0, else 0
    grad = -np.dot(mask * y, x) / N
    return grad

print("Initial gradient:", grad_HL(X_train, y_train, w))

# Gradient Descent 
alpha      = 0.01        
iterations = 1000

print("Initial Hinge Loss:", hinge_Loss(X_train, y_train, w))

for i in range(iterations):
    grad = grad_HL(X_train, y_train, w)
    w    = w - alpha * grad
    loss = hinge_Loss(X_train, y_train, w)
    if (i + 1) % 100 == 0:
        print(f"Iteration {i+1:4d} | Loss: {loss:.6f}")

print("\nOptimal weights:", w)
print("Minimum Hinge Loss:", hinge_Loss(X_train, y_train, w))

# Predict on test set 
y_pred = np.sign(z(X_test, w))        # -1 or +1
y_pred = np.where(y_pred == 0, -1, y_pred)   # edge-case: sign(0) → -1

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Benign (-1)", "Malignant (1)"]))
print("Accuracy:", accuracy_score(y_test, y_pred))


#                        OPEN-ENDED QUESTION:
#       How do you evaluate the performance of an applied machine learning model? 
#  Which evaluation metrics are most important for the correct implementation of a given task, and why?


# To evaluate a machine learning model, we check how well it predicts outcomes on data it has never seen before 
# (the test set). We don't just look at one number — we use several metrics together to get the full picture.
# For this task, the model classifies tumors as either benign (-1) or malignant (+1). 
# The most important metric here is Recall (also called Sensitivity). 
# Recall tells us: out of all the actual malignant tumors, how many did the model correctly catch?
# This matters most here because the consequences of errors are not equal:
# If the model says a tumor is benign when it's actually malignant (false negative) 
# → the patient doesn't get treatment → this can be life-threatening.
# If the model says a tumor is malignant when it's actually benign (false positive) 
# → the patient gets extra tests → this is unpleasant but not dangerous.

# So we want to minimize missed malignant cases at all costs, which means maximizing recall for the malignant class.
# We also look at:

# Accuracy — overall percentage of correct predictions (but can be misleading if one class is more common)
# Precision — how many of the predicted malignant cases are actually malignant
# Recall — out of all actually good applicants, how many did the model correctly approved
# F1-score — the balance between precision and recall
# Confusion matrix — a table showing exactly how many cases were correctly or incorrectly classified

# In summary,for a medical diagnosis task like this one,recall is the priority because missing a cancer is far worse than a false alarm.