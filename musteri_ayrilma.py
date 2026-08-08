"""
Ödev Amaç: 
Sentetik 1000 satırlık veri setinden yararlanarak müşteri ayrılma tahmini tespiti yapılması

Kullanılan Kütüphaneler:
Pandas
Scikit Learn


Gerçek Hayat Örneği:
Müşteri Ayrılma Tahmini Modeli
EC Hosting firmasına ait veri setinden yola çıkarak hosting firmalarındaki müşterilerin ayrılma analizi

Adımlar:
[+] 1. Python dosyanızın başına docstring ekleyin. Docstring içinde ödevin amacını, kullanılan kütüphaneleri ve çalıştırma adımlarını kısaca açıklayın.
[+] 2. Veri setini pandas DataFrame olarak hazırlayın veya CSV dosyasından okuyun.
[+] 3. Veri setinin ilk satırlarını, satır-sütun sayısını ve hedef değişken dağılımını inceleyin.
[+] 4.1 Eksik değer kontrolü yapın. Eksik değer varsa uygun şekilde doldurun veya temizleyin.
[+] 4.2 Aykırı değer kontrolü yapın. Aykırı değer varsa uygun şekilde temizleyin.
[+] 5. Kategorik değişkenleri One-Hot Encoding veya uygun bir yöntemle sayısal forma dönüştürün.
[+] 6. Sayısal değişkenlerde gerekli gördüğünüz yerlerde ölçekleme uygulayın.
[+] 7. En az 1 tane basit öznitelik üretin. Örnek: gelir_grubu, destek_talebi_var_mi, abonelik_yili veya benzer anlamlı bir değişken.
[+] 8. Veriyi train, validation ve test kümelerine ayırın. Sınıflandırma problemi olduğu için mümkünse stratify kullanın.
[+] 9. En az 2 model eğitin. Önerilen modeller: Logistic Regression ve KNN. İsterseniz Decision Tree modelini bonus olarak ekleyebilirsiniz.
[+] 10. Validation sonuçlarına göre modelleri karşılaştırın ve seçtiğiniz modeli test verisi üzerinde değerlendirin.
[+] 11. Test seti için confusion matrix, accuracy, precision, recall ve F1-score değerlerini yazdırın.
[+] 12. Kodun sonunda kısa bir yorum çıktısı üretin: Hangi model daha iyi oldu? Sizce neden?
"""
import pandas as pd

from sklearn.model_selection import train_test_split # eğitim ve test veri seti oluşturur
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("musteri_ayrilma_veriseti_yeni.csv")

print(df.head(10))
print(df.tail(10))

print(df.shape) # (1000, 8) 1000 satır 8 sütun(7 data + 1 churn)
print(df.info())

# Adet bazında dağılım
dagilim_adet = df['Churn'].value_counts()
print("Adet Bazında Dağılım:\n", dagilim_adet)

# Yüzde bazında dağılım (normalize=True parametresi ile)
dagilim_yuzde = df['Churn'].value_counts(normalize=True) * 100
print("\nYüzde Bazında Dağılım:\n", dagilim_yuzde)

print(df.isnull().sum())

df_filled = df.copy()

sayisal_sutunlar = list(df_filled.select_dtypes(include=['int64', 'float64']).columns)
doldurulmasi_gereken_sayisal_sutunlar = ["Yas","Aylik_Harcama","Destek_Talebi_Sayisi"]
for sutun in doldurulmasi_gereken_sayisal_sutunlar:
    df_filled[sutun] = df_filled[sutun].fillna(df_filled[sutun].median())
print(df_filled.isnull().sum())

doldurulmasi_gereken_kategorik_sutunlar = ["Sehir","Uyelik_Tipi"]
for sutun in doldurulmasi_gereken_kategorik_sutunlar:
    df_filled[sutun] = df_filled[sutun].fillna(df_filled[sutun].mode()[0])
print(df_filled.isnull().sum())

df_filled_checked = df_filled.copy()

# 4. IQR yöntemiyle aykırı değerleri tespit etme Çok karmaşık geldiği için kendim yapamadım ama mantığını anlayarak yaptım

aykiri_deger_maskesi = pd.Series(False, index = df_filled.index)

for sutun in sayisal_sutunlar:

    q1 = df_filled[sutun].quantile(0.25)
    q3 = df_filled[sutun].quantile(0.75)

    iqr = q3 - q1

    alt_sinir = q1 - 2 * iqr # 1.5 olunca çok değer çıkardı o yüzden 2 yaptım
    ust_sinir = q3 + 2 * iqr

    sutun_maskesi = (
        (df_filled[sutun] < alt_sinir) | (df_filled[sutun] > ust_sinir)
    )

    aykiri_deger_maskesi = aykiri_deger_maskesi | sutun_maskesi

    print(f"Aykırı değer sayısı: {sutun_maskesi.sum()}")

    if sutun_maskesi.any():
        print(f"Aykırı değerler: \n{df_filled.loc[sutun_maskesi, sutun]}")

print(f"En az bir aykırı değer içeren satırlar \n{df_filled.loc[aykiri_deger_maskesi]}")

# aykırı değer içeren satırları veri setinden çıkartalım
df_clean = df_filled.loc[~aykiri_deger_maskesi].copy()
df_clean.reset_index(drop=True, inplace=True)

print(f"Aykırı değerler çıktıktan sonra \n{df_clean}")

df_clean.to_csv("musteri_ayrilma_veriseti_yeni_duzenlendi.csv", index=False)

df = pd.read_csv("musteri_ayrilma_veriseti_yeni_duzenlendi.csv")

label_encoder = LabelEncoder()


# Üyelik tipleri için kendi mantıksal sıralamamızı (hiyerarşiyi) kuruyoruz
uyelik_sozlugu = {
    "Temel": 0,
    "Standart": 1,
    "Premium": 2
}
df["Uyelik_Tipi"] = df["Uyelik_Tipi"].map(uyelik_sozlugu)

df = pd.get_dummies(df, columns=["Sehir"], drop_first=True, dtype=int)
#print(df)



df['Destek_Talebi_Var_Mi'] = (df['Destek_Talebi_Sayisi'] > 0).astype(int)
df['Sadik_Musteri'] = (df['Abonelik_Suresi_Ay'] >= 12).astype(int)
df['Toplam_Kazanc'] = df['Aylik_Harcama'] * df['Abonelik_Suresi_Ay']
df['Abonelik_Yili'] = round(df['Abonelik_Suresi_Ay'] / 12, 1)






df.to_csv("musteri_ayrilma_veriseti_yeni_duzenlendi_son.csv", index=False)
# print(df)



df = pd.read_csv("musteri_ayrilma_veriseti_yeni_duzenlendi_son.csv")


X = df.drop(columns=["Musteri_ID","Churn"])
y = df["Churn"]





# Train test split konusunda yapay zekadan yardım aldım çünkü 70/15/15 ayırmaya çalışırken hata yaptım
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # val = %80, test = %20

# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.4, random_state=42, stratify=y_train_val)

# Verinin %70'ini Eğitim, %30'unu (Geçici) Test+Val seti olarak ayırıyoruz.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Böylece %15 Validation ve %15 Test elde etmiş oluyoruz.
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)


sayisal_sutunlar = ['Yas', 'Aylik_Harcama', 'Abonelik_Suresi_Ay', 'Destek_Talebi_Sayisi', 'Toplam_Kazanc', 'Abonelik_Yili']
standard_scaler = StandardScaler()

X_train_standard = X_train.copy()
X_val_standard = X_val.copy()
X_test_standard = X_test.copy()

X_train_standard[sayisal_sutunlar] = (
    standard_scaler.fit_transform(
        X_train[sayisal_sutunlar]
    )
)

X_val_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_val[sayisal_sutunlar]
    )
)

X_test_standard[sayisal_sutunlar] = (
    standard_scaler.transform(
        X_test[sayisal_sutunlar]
    )
)












# print(f"Toplam Veri Sayısı: {len(X)}")
# print(f"Eğitim (Train) Seti: {len(X_train)} satır")
# print(f"Doğrulama (Validation) Seti: {len(X_val)} satır")
# print(f"Test Seti: {len(X_test)} satır")


logistic_reg = LogisticRegression(l1_ratio=0, C = 0.1, max_iter = 100) # class_weight='balanced'
logistic_reg.fit(X_train_standard, y_train)

# y_pred = logistic_reg.predict(X_test_standard)
# logistic_reg_acc = accuracy_score(y_test, y_pred)
# print(f"Logistic Regression Accuracy: {logistic_reg_acc}")
# logistic_reg_conf_matrix = confusion_matrix(y_test, y_pred)
# print(f"Logistic Regression Confusion Matrix: \n{logistic_reg_conf_matrix}")

# logistic_reg_acc = logistic_reg.score(X_test_standard, y_test)
# print(f"Logistic Regression Accuracy: {logistic_reg_acc}")

# Logistic Regression Accuracy: 0.6736111111111112
# Logistic Regression Confusion Matrix: 
# [[87  7]
#  [40 10]]




knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_standard, y_train)


# y_pred = knn.predict(X_test_standard)

# knn_accuracy = accuracy_score(y_test, y_pred)
# print(f"KNN Accuracy: {knn_accuracy}")

# knn_conf_matrix = confusion_matrix(y_test, y_pred)
# print(f"KNN Confusion Matrix: \n{knn_conf_matrix}")

# KNN Accuracy: 0.6388888888888888
# KNN Confusion Matrix: 
# [[78 16]
#  [36 14]]



tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree_clf.fit(X_train_standard, y_train)

# y_pred = tree_clf.predict(X_test_standard)

# tree_clf_accuracy = accuracy_score(y_test, y_pred)
# print(f"Decision Tree Accuracy: {tree_clf_accuracy}")

# tree_clf_conf_matrix = confusion_matrix(y_test, y_pred)
# print(f"Decision Tree Confusion Matrix: \n{tree_clf_conf_matrix}")

# Decision Tree Accuracy: 0.6458333333333334
# Decision Tree Confusion Matrix: 
# [[87  7]
#  [44  6]] 


logistic_reg_y_pred = logistic_reg.predict(X_test_standard)
knn_y_pred = knn.predict(X_test_standard)
tree_y_pred = tree_clf.predict(X_test_standard)

logistic_reg_acc = accuracy_score(y_test, logistic_reg_y_pred)
print(f"Logistic Regression Accuracy: {logistic_reg_acc}")
logistic_reg_conf_matrix = confusion_matrix(y_test, logistic_reg_y_pred)
print(f"Logistic Regression Confusion Matrix: \n{logistic_reg_conf_matrix}")
print("Logistic Regression Classification Report:")
print(classification_report(y_test, logistic_reg_y_pred))
print("-" * 40)

knn_accuracy = accuracy_score(y_test, knn_y_pred)
print(f"KNN Accuracy: {knn_accuracy}")
knn_conf_matrix = confusion_matrix(y_test, knn_y_pred)
print(f"KNN Confusion Matrix: \n{knn_conf_matrix}")
print("KNN Classification Report:")
print(classification_report(y_test, knn_y_pred))
print("-" * 40)

tree_clf_accuracy = accuracy_score(y_test, tree_y_pred)
print(f"Decision Tree Accuracy: {tree_clf_accuracy}")
tree_clf_conf_matrix = confusion_matrix(y_test, tree_y_pred)
print(f"Decision Tree Confusion Matrix: \n{tree_clf_conf_matrix}")
print("KNN Classification Report:")
print(classification_report(y_test, tree_y_pred))


"""
Tüm sonuçlara göre bu veri setine en uygun model Logistic Regression'dir
Bence en iyi model olma sebebi lasso cezalandırma olmasıdır.

SONUÇLAR:
Logistic Regression Accuracy: 0.6736111111111112
Logistic Regression Confusion Matrix: 
[[87  7]
 [40 10]]
Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       0.69      0.93      0.79        94
           1       0.59      0.20      0.30        50

    accuracy                           0.67       144
   macro avg       0.64      0.56      0.54       144
weighted avg       0.65      0.67      0.62       144

----------------------------------------
KNN Accuracy: 0.6388888888888888
KNN Confusion Matrix: 
[[78 16]
 [36 14]]
KNN Classification Report:
              precision    recall  f1-score   support

           0       0.68      0.83      0.75        94
           1       0.47      0.28      0.35        50

    accuracy                           0.64       144
   macro avg       0.58      0.55      0.55       144
weighted avg       0.61      0.64      0.61       144

----------------------------------------
Decision Tree Accuracy: 0.6458333333333334
Decision Tree Confusion Matrix: 
[[87  7]
 [44  6]]
KNN Classification Report:
              precision    recall  f1-score   support

           0       0.66      0.93      0.77        94
           1       0.46      0.12      0.19        50

    accuracy                           0.65       144
   macro avg       0.56      0.52      0.48       144
weighted avg       0.59      0.65      0.57       144


"""