from ucimlrepo import fetch_ucirepo

concrete_compressive_strength = fetch_ucirepo(id=165)

X = concrete_compressive_strength.data.features
y = concrete_compressive_strength.data.targets

print("Shape:", X.shape)
print("\nMissing values in X:\n", X.isnull().sum())
print("\nMissing values in y:\n", y.isnull().sum())
print("\nObject-type columns in X:\n", X.select_dtypes(include='object').columns.tolist())

# Impute missing values with column mean (mean imputation)
import pandas as pd
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Convert any object-type columns to numeric 
for col in X.select_dtypes(include='object').columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.mean())

print("\nDataset is clean. Shape:", X.shape)

# Scale features (StandardScaler) to improve gradient descent convergence
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

#  Convert to numpy arrays & add bias column
import numpy as np

# Add a column of ones at the beginning for the bias term (b) for computing w·x + b as a single dot product
X = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])
y = y.values.flatten()   

print("X shape (with bias col):", X.shape)
print("y shape:", y.shape)

# w (weights vector) has size = number of features + 1 (the +1 is for the bias term)
w = np.zeros(X.shape[1])   
print("Initial weights:", w)

#  Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

#  z function — linear combination z = X @ w 
def z(x, w):
    return x @ w   

#  Squared Loss (MSE) function
def squared_Loss(x, y, w):
    n = x.shape[0]
    predictions = z(x, w)
    errors = predictions - y
    return (1 / (2 * n)) * np.dot(errors, errors)

print("Initial loss:", squared_Loss(X_train, y_train, w))

#  Gradient of Squared Loss
def grad_SL(x, y, w):
    n = x.shape[0]
    errors = z(x, w) - y         
    return (1 / n) * (x.T @ errors) 

print("Initial gradient:", grad_SL(X_train, y_train, w))

#  Gradient Descent
alpha = 0.01        # learning rate
iterations = 1000  # number of iterations

w = np.zeros(X_train.shape[1])

print(f"Initial loss: {squared_Loss(X_train, y_train, w):.4f}")

loss_history = []

for i in range(iterations):
    grad = grad_SL(X_train, y_train, w)   
    w = w - alpha * grad                  
    loss = squared_Loss(X_train, y_train, w)  
    loss_history.append(loss)

    if i % 100 == 0:
        print(f"  Iteration {i:4d} | Loss: {loss:.4f}")

print(f"\nFinal loss:    {loss_history[-1]:.4f}")
print(f"Optimal weights: {w}")

#  Evaluate on test set
from sklearn.metrics import r2_score, mean_squared_error

y_pred = z(X_test, w)   

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nR2_score:  {r2:.4f}")
print(f"RMSE:      {rmse:.4f}")

# Plot loss curve over iterations
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("Squared Loss")
plt.title("Gradient Descent — Loss over Iterations")
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()

# Polynomial Regression
# Skip the bias weight (index 0); feature weights are indices 1..8
feature_names = concrete_compressive_strength.data.features.columns.tolist()
feature_weights = np.abs(w[1:])   
most_important_idx = np.argmax(feature_weights)
most_important_feature = feature_names[most_important_idx]

print(f"\nMost important feature: '{most_important_feature}' "
      f"(|weight| = {feature_weights[most_important_idx]:.4f})")

# Polynomial Regression from degree 2 to 8 on the most important feature
X_raw = concrete_compressive_strength.data.features.values
x_feat = X_raw[:, most_important_idx]  

# Scale that single feature
feat_scaler = StandardScaler()
x_feat_scaled = feat_scaler.fit_transform(x_feat.reshape(-1, 1)).flatten()

degrees = range(2, 9)
r2_train_list, r2_test_list = [], []

plt.figure(figsize=(14, 10))

for deg in degrees:
    # Build polynomial feature matrix [1, x, x^2, ..., x^deg]
    poly_cols = [x_feat_scaled ** d for d in range(deg + 1)]
    X_poly = np.column_stack(poly_cols)   
    X_poly = np.clip(X_poly, -10, 10)

    # Train/test split 
    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )

    # Gradient descent for this polynomial model
    w_poly = np.zeros(X_poly.shape[1])
    alpha_poly = 0.001
    iters_poly = 4000

    for _ in range(iters_poly):
        g = grad_SL(Xp_train, yp_train, w_poly)
        w_poly = w_poly - alpha_poly * g

    yp_pred_train = z(Xp_train, w_poly)
    yp_pred_test  = z(Xp_test,  w_poly)

    r2_tr = r2_score(yp_train, yp_pred_train)
    r2_te = r2_score(yp_test,  yp_pred_test)
    r2_train_list.append(r2_tr)
    r2_test_list.append(r2_te)

    print(f"Degree {deg}: Train R²={r2_tr:.4f}  |  Test R²={r2_te:.4f}")

    ax = plt.subplot(3, 3, deg - 1)
    x_sort = np.sort(x_feat_scaled)
    X_line = np.column_stack([x_sort ** d for d in range(deg + 1)])
    y_line = z(X_line, w_poly)

    ax.scatter(x_feat_scaled, y, alpha=0.3, s=10, label="Data")
    ax.plot(x_sort, y_line, color="red", linewidth=2, label=f"Deg {deg}")
    ax.set_title(f"Degree {deg}  |  R²={r2_te:.3f}")
    ax.set_xlabel(most_important_feature)
    ax.set_ylabel("Strength")
    ax.legend(fontsize=7)

plt.suptitle(f"Polynomial Regression (feature: {most_important_feature})", fontsize=13)
plt.tight_layout()
plt.savefig("poly_regression_curves.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(list(degrees), r2_train_list, marker='o', label="Train R²")
plt.plot(list(degrees), r2_test_list,  marker='s', label="Test R²")
plt.xlabel("Polynomial Degree")
plt.ylabel("R² Score")
plt.title("Underfitting vs Overfitting")
plt.xticks(list(degrees))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("r2_vs_degree.png", dpi=150)
plt.show()