import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

df = pd.read_csv('Coffee_sales.csv')

#print(f"Размер датасета: {df.shape}")
#print(df['coffee_name'].value_counts())

df_encoded = df.copy()
encoder = LabelEncoder()
df_encoded['coffee_encode'] = encoder.fit_transform(df['coffee_name'])

#print("\nПосле кодирования:")
#print(df_encoded['coffee_encode'].value_counts())

features = ['hour_of_day', 'money', 'Weekdaysort', 'Monthsort']
x = df_encoded[features]
y = df_encoded['coffee_encode']
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)
scaler = StandardScaler()
x_training_scaled = scaler.fit_transform(x_training_data)
x_test_scaled = scaler.transform(x_test_data)

models = {
    #Случайный лес 
    'Random Forest': RandomForestClassifier(random_state=42),
    #К-ближайших соседей 
    'K-Neighbors': KNeighborsClassifier(n_neighbors = 1),
    #Дерево решений 
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    #Адаптивный бустинг 
    'AdaBoost' : AdaBoostClassifier(n_estimators=100, random_state=42, learning_rate=0.5),
    #Наивный Байес 
    'GaussianNB' : GaussianNB()
}

results = {}

print("\nРезультаты: ")
for name, model in models.items():
    model.fit(x_training_scaled, y_training_data)   
    predictions = model.predict(x_test_scaled) 
    accuracy = accuracy_score(y_test_data, predictions)
    results[name] = accuracy
    
    print(f"\n{name}")
    print(f"Accuracy: {accuracy:.5f}")
    f1 = f1_score(y_test_data, predictions, average='weighted')
    print(f"F1-Score: {f1:.5f}")
    precision = precision_score(y_test_data, predictions, average='weighted', zero_division=0)
    print(f"Precision: {precision:.5f}")
    recall = recall_score(y_test_data, predictions, average='weighted')
    print(f"Recall: {recall:.5f}")

#График 1: Сравнение точности предсказаний (accurancy) 
plt.figure(figsize=(8, 6))
bars = plt.bar(results.keys(), results.values(),
               color=['#228B22', '#FF8C00', '#66CDAA', '#DC143C', '#FFD700'])
plt.title('Сравнение точности моделей', fontsize=14, fontweight='bold')
plt.ylabel('Точность')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', ha='center', va='bottom')

#График 2: Корреляция признаков с переменной coffee_encode 
plt.figure(figsize=(10, 6))
corr_matrix = df_encoded[features + ['coffee_encode']].corr()
corr_fixed = corr_matrix['coffee_encode'].drop('coffee_encode')
plt.barh(corr_fixed.index, corr_fixed.values, color='#9370D8')
plt.title('Корреляция признаков с целевой переменной coffee_encode', fontweight='bold')
plt.xlabel('Значение коэффициента корреляции')
plt.xlim(-1, 1)

for index, value in enumerate(corr_fixed.values):
    plt.text(value, index, f'{value:.3f}', ha='left', va='center')
    
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.show()

