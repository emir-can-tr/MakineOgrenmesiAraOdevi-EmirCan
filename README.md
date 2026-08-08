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
<img width="467" height="774" alt="image" src="https://github.com/user-attachments/assets/9cd96ecd-648f-46a9-85d3-af085c824453" />



Developed by [EC](https://emircan.tr)
