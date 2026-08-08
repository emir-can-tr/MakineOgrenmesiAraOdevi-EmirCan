# Makine Öğrenmesi Ara Ödev – Emir Can - Müşteri Ayrılma Tahmini Projesi

## Projenin Amaçı

Sentetik 1000 sample ve beraberinde eksik/aykırı veri içeren veri setinden "Müşteri Ayrılma Tahmini Modeli" geliştirilmesi

## Nasıl Çalıştırılır?

1. Veri setinizi hazırlayın. Adı "musteri_ayrilma_veriseti_yeni.csv" olmalıdır.
2. ```Shell
   python -m venv venv
   ```
3. ```Shell
   .\venv\Scripts\activate
   ```
4. ```Shell
   pip install -r requirements.txt
   ```
5. ```Shell
   python musteri_ayrilma.py
   ```

## Sonuç

Tüm sonuçlara göre bu veri setine en uygun model Logistic Regression'dir
Bence en iyi model olma sebebi lasso cezalandırma olmasıdır.

SONUÇLAR:
Logistic Regression Accuracy: 0.6736111111111112
Logistic Regression Confusion Matrix:
[[87  7]
 [40 10]]
Logistic Regression Classification Report:
precision recall f1-score support

    0       0.69      0.93      0.79        94
           1       0.59      0.20      0.30        50

    accuracy                           0.67       144

macro avg 0.64 0.56 0.54 144
weighted avg 0.65 0.67 0.62 144

---

KNN Accuracy: 0.6388888888888888
KNN Confusion Matrix:
[[78 16]
 [36 14]]
KNN Classification Report:
precision recall f1-score support

    0       0.68      0.83      0.75        94
           1       0.47      0.28      0.35        50

    accuracy                           0.64       144

macro avg 0.58 0.55 0.55 144
weighted avg 0.61 0.64 0.61 144

---

Decision Tree Accuracy: 0.6458333333333334
Decision Tree Confusion Matrix:
[[87  7]
 [44  6]]
KNN Classification Report:
precision recall f1-score support

    0       0.66      0.93      0.77        94
           1       0.46      0.12      0.19        50

    accuracy                           0.65       144

macro avg 0.56 0.52 0.48 144
weighted avg 0.59 0.65 0.57 144

Developed by [EC](https://emircan.tr)
