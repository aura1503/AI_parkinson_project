import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def generate_combinations(base_neurons, learning_rates):
    combinations = []
    neuron_multipliers = [0.5, 1, 2]

    for n_layers in [1, 2]:
        for first_multiplier in neuron_multipliers:
            first_neurons = int(base_neurons * first_multiplier)

            if n_layers == 1:
                for lr in learning_rates:
                    combinations.append({
                        'hidden_layers': (first_neurons,),
                        'learning_rate': lr
                    })
            else:
                for second_multiplier in neuron_multipliers:
                    second_neurons = int(first_neurons * second_multiplier)
                    for lr in learning_rates:
                        combinations.append({
                            'hidden_layers': (first_neurons, second_neurons),
                            'learning_rate': lr
                        })
    return combinations

df = pd.read_csv("parkinsons.data")

X = df.drop(columns=['name', 'status'])
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

learning_rates = [0.1, 0.01]
base_neurons = X.shape[1]
combinations = generate_combinations(base_neurons, learning_rates)

results = []

for i, combo in enumerate(combinations):
    hidden_layers = combo['hidden_layers']
    lr = combo['learning_rate']

    print(f"\nConfiguration {i}: Hidden Layers: {hidden_layers}, Learning Rate: {lr}")

    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, learning_rate_init=lr, max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy Score: {acc:.4f}")
    print("Classification Report:")
    print(clf_report)
    print("Confusion Matrix:")
    print(conf_matrix)
    print('\n*****************************************************************')

    results.append({
        'Configuration': f'Hidden Layers: {hidden_layers}, Learning Rate: {lr}',
        'Accuracy': acc
    })

results_df = pd.DataFrame(results)
print("\nSummary of all configurations:")
print(results_df.sort_values(by='Accuracy', ascending=False))

