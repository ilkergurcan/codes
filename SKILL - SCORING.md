---
name: skorlama-setup
description: >
  ML projeleri için skorlama/batch predict pipeline kurulumunu adım adım yürütür.
  config.py, batch_predict.py, main.py ve values.yaml dosyalarını doldurur,
  test ve prod ortamı geçişlerini yönetir.
  Bu skill şu durumlarda tetiklenir: skorlama kurulumu, batch predict,
  skorlama pipeline, model skorlama, "skorlamayı kur", "batch predict hazırla",
  "modeli skorla", "skorlama pipeline'ı başlat" gibi ifadeler.
---

# Skorlama / Batch Predict Pipeline Kurulum Skill'i

Bu skill, ML projelerinin skorlama pipeline'ını baştan sona kurar.
Test ve prod ortamı geçişlerini yönetir.
Kullanıcıya her adımı sırayla sor, bir adım tamamlanmadan diğerine geçme.

## Genel Kurallar

- Her adımı sırayla takip et, atlama yapma.
- Kullanıcıdan bilgi beklerken, kullanıcı bir **hata** paylaşırsa:
  1. Önce hatayı analiz et ve çözüm öner.
  2. Hata çözüldükten sonra kaldığın adıma geri dön ve eksik bilgiyi tekrar iste.
  3. Kullanıcıya hangi adımda olduğunu hatırlat.
- Dosya düzenlerken şablon yapısını **kesinlikle bozma**. Şablon yapısını anlamak
  için bu skill klasöründeki referans dosyaları kullan.
- Her dosya değişikliğinden önce mevcut dosyayı oku ve yapıyı anla.
- Türkçe iletişim kur, teknik terimleri olduğu gibi bırak (git, branch, CI, deploy vb.).
- **Batch klasöründe yapılan HER değişiklikten sonra**, son commit'in ID'sini
  `values.yaml`'daki `ImageId` alanına yazmayı unutma. Bu kural tüm adımlar
  boyunca geçerlidir.

## Şablon Referans Dosyaları

Bu skill klasöründe şu referans dosyalar bulunur:

- **`config_template.py`** — config.py'nin orijinal şablon hali.
- **`batch_predict_template.py`** — batch_predict.py'nin orijinal şablon hali.
- **`main_template.py`** — main.py'nin orijinal şablon hali.
  SQL script çalıştırma alanı ve `execute_file` fonksiyon kullanımı burada belirtilmiştir.
- **`values-example.yaml`** — values.yaml'ın örnek yapısı.

**Her dosya düzenlemesinden önce** ilgili referans dosyayı oku ve mevcut dosyayla
karşılaştır. Düzenleme yaparken referanstaki yapıyı baz al.

---

## Adım 1: Remote Senkronizasyonu

Remote'taki değişiklikleri al:
```bash
git pull --rebase origin master
```

Eğer rebase sırasında conflict çıkarsa:
1. Conflict olan dosyaları kullanıcıya göster.
2. Çözüm öner ve uygula.
3. Rebase'i tamamla:
```bash
git add .
git rebase --continue
```

Başarılı olduktan sonra Adım 2'ye geç.

---

## Adım 2: config.py Bilgilerini Topla

`Batch/src/config.py` dosyasını oku ve bu skill klasöründeki
`config_template.py` referans dosyasıyla karşılaştır.

Kullanıcıya **sırayla** şunları sor:

1. **MODEL_ID:**
   "Model ID'yi paylaşır mısın? Birden fazla model varsa hepsini virgülle
   ayırarak yazabilirsin. (Örnek: 101 veya 101, 102, 103)"

2. **PRODUCT_CODE:**
   "Ürün kodunu paylaşır mısın? Birden fazlaysa virgülle ayır.
   (Örnek: KKSA veya KKSA, KMH, ETKRD)"

3. **CODE:**
   "Model kodunu paylaşır mısın?"

4. **REASON1 (Sebep):**
   "Modelin sebebini belirtir misin? Örnek olarak şunlardan birini
   seçebilirsin: aktif, inaktif, neveraktif"

5. **CALIBRATION:**
   "Model kalibrasyon ile mi çalışıyor?
   - Evet → 1
   - Hayır → 0"

**Birden fazla model durumu:**
Eğer MODEL_ID veya PRODUCT_CODE birden fazla değer içeriyorsa, bunları
Python listesi formatında config.py'a yaz:
```python
MODEL_ID = [101, 102, 103]
PRODUCT_CODE = ["KKSA", "KMH", "ETKRD"]
```

Tek değer varsa:
```python
MODEL_ID = 101
PRODUCT_CODE = "KKSA"
```

---

## Adım 3: config.py Güncelle

Adım 2'de toplanan bilgilerle `Batch/src/config.py` dosyasını güncelle.
Referans şablondaki yapıyı koru, sadece ilgili alanları doldur.

Değişikliği kullanıcıya göster ve onayla:
"config.py şu şekilde güncellendi, uygun mu?"

Onay alındıktan sonra Adım 4'e geç.

---

## Adım 4: batch_predict.py Dosyasını Doldur

### Aşama 4a: Kaynak Notebook'u Bul

Repo içinde `SKORLAMA` veya `skorlama` içeren notebook ara:
```bash
find . -name "SKORLAMA*-*.ipynb" -o -name "SKORLAMA -*".ipynb -o -name "skorlama*-*.ipynb" -o -name "skorlama -*".ipynb 2>/dev/null
```

**Eğer notebook bulunamazsa**, kullanıcıya doğrudan sor:
"Skorlama notebook'unu bulamadım. Skorlama/batch predict kodunu
içeren notebook veya Python dosyasının tam yolunu paylaşır mısın?"

Eğer birden fazla notebook bulunursa, listeyi kullanıcıya göster ve
hangisini kullanacağını sor.

### Aşama 4b: Skorlama Kodunu batch_predict.py'ye Aktar

1. Bu skill klasöründeki `batch_predict_template.py` referans dosyasını oku.
2. Repodaki `Batch/src/batch_predict.py` dosyasını oku.
3. Referans dosyadaki korunan ve düzenlenecek bölümleri tespit et.

**Korunan bölümler (kesinlikle değiştirme):**
- `# === BU BÖLÜME DOKUNMA ===` ile işaretlenmiş bloklar.
- Model yükleme ve data okuma mekanizmaları.
- Model kaydetme/çıktı yazma bölümleri.

**Notebook'tan aktarılacak kod:**
- Feature engineering / preprocessing adımları
- Skorlama (predict / predict_proba) mantığı
- Post-processing adımları

**Notebook'tan aktarılMAyacak kod:**
- EDA, görselleştirme, debug hücreleri
- Model eğitim (fit) kodu — bu CI'da zaten yapıldı
- Manuel dosya okuma/yazma kodları

Düzenleme tamamlandığında dosyayı kullanıcıya göster:
"batch_predict.py şu şekilde güncellendi. Korunan bölümlere
dokunmadım. Uygun mu?"

Onay alındıktan sonra Adım 5'e geç.

---

## Adım 5: SQL Sorgularını Topla ve Queries Klasörüne Koy

Kullanıcıya sor:
"Skorlama için SP (stored procedure), müşteri datası oluşturma sorgusu
veya skorlama sorgusu var mı? Varsa aşağıdaki yollardan biriyle paylaşabilirsin:
- `.sql` uzantılı dosya adı/yolu (birden fazla olabilir)
- SQL sorgusunu doğrudan chat'e yapıştır"

### Dosya olarak verilirse:

Her `.sql` dosyasını **ismini değiştirmeden** `Batch/src/Queries/` klasörüne kopyala:
```bash
cp <DOSYA_YOLU> Batch/src/Queries/
```

### Chat'e yapıştırılırsa:

Kullanıcıya sor:
"Bu sorgu için bir dosya adı belirleyelim. (Örnek: musteri_kitle.sql)"

Sonra sorguyu verilen isimle kaydet:
```bash
cat > Batch/src/Queries/<DOSYA_ADI> << 'EOF'
<SQL_SORGUSU>
EOF
```

### Queries klasörü kontrolü:

`Batch/src/Queries/` klasöründe `example_query.sql`, `ornek_sorgu.sql` gibi
örnek dosyalar varsa bunları **görmezden gel** — main.py'ye ekleme.

Kullanıcıya eklenen dosyaları listele ve onayla:
"Şu SQL dosyaları Queries klasörüne eklendi: [dosya listesi]. Doğru mu?"

Onay alındıktan sonra Adım 6'ya geç.

---

## Adım 6: main.py Dosyasını Güncelle

1. Bu skill klasöründeki `main_template.py` referans dosyasını oku.
2. Repodaki `Batch/main.py` dosyasını oku.
3. SQL script çalıştırma alanını tespit et. Bu alan genellikle
   **265. ve 271. satırlar** arasında bulunur, ancak `main_template.py`'deki
   işaretlemeleri baz al.

### SQL Dosyalarını main.py'ye Ekle

Adım 5'te `Batch/src/Queries/` klasörüne koyulan her `.sql` dosyası için
bir `query_path` tanımla ve `execute_file` fonksiyonu ile çalıştır.

**Tek SQL dosyası varsa:**
```python
query_path = os.path.join(SRC_DIR, "Queries", "<DOSYA_ADI>.sql")
exe.execute_file(query_path, show_sql=False)
```

**Birden fazla SQL dosyası varsa:**
Her biri için ayrı `query_path` oluştur ve sırayla çalıştır:
```python
query_path_1 = os.path.join(SRC_DIR, "Queries", "<DOSYA_1>.sql")
exe.execute_file(query_path_1, show_sql=False)

query_path_2 = os.path.join(SRC_DIR, "Queries", "<DOSYA_2>.sql")
exe.execute_file(query_path_2, show_sql=False)
```

**Önemli:** `example_query.sql` gibi örnek dosyaları ekleme.

Düzenleme tamamlandığında dosyayı kullanıcıya göster ve onayla.

---

## Adım 7: requirements.txt Güncelle

`Batch/` klasöründeki (veya repo kökündeki) `requirements.txt` dosyasını oku.

batch_predict.py ve main.py'de kullanılan tüm kütüphaneleri kontrol et.
Eksik olan kütüphaneleri `requirements.txt`'ye ekle.

Yaygın eklenmesi gereken kütüphaneler:
- `dill` (model yükleme için)
- `pyodbc` (veritabanı bağlantısı)
- `pandas`, `numpy` (veri işleme)
- Modele özel kütüphaneler (lightgbm, xgboost, catboost vb.)

Mevcut kütüphaneleri silme, sadece eksikleri ekle.
Değişikliği kullanıcıya göster ve onayla.

---

## Adım 8: Test Aşamasına Hazırlık — Kullanıcıyı Bilgilendir

Kullanıcıya şunu söyle:
"Kod tarafı hazır. Şimdi önce **test ortamında** modelin düzgün çalıştığını
kontrol edeceğiz. Test sonuçlarını doğrulamak için aşağıdaki SQL sorgusunu
kullanabilirsin:"

**Not:** Bu SQL sorgusu henüz skill dosyasına eklenmemiştir.
Kullanıcıya şu mesajı ver:
"Test doğrulama SQL sorgusu yakında bu skill'e eklenecek.
Şimdilik test sonuçlarını OpenShift üzerinden ve çıktı tablosunu
sorgulayarak doğrulayabilirsin."

<!-- TODO: Test doğrulama SQL sorgusunu buraya ekle -->

Sonra Adım 9'a geç.

---

## Adım 9: values.yaml Test Konfigürasyonu

`Projects/MainProject/values.yaml` dosyasını oku ve bu skill klasöründeki
`values-example.yaml` ile karşılaştır.

Aşağıdaki alanları **sırayla** güncelle:

### 9.1: ImageId

Batch klasöründeki değişikliklerin commit ID'sini al:
```bash
git add Batch/
git commit -m "Skorlama pipeline konfigürasyonu tamamlandı"
git log -1 --format="%H" -- Batch/
```
Bu commit ID'yi `values.yaml`'daki `ImageId` alanına yaz.

### 9.2: cronnExpression (Test)

Test aşamasında 5 dakikada bir çalıştırılması tavsiye edilir.
Kullanıcıya bilgi ver:
"Test aşaması için cron'u 5 dakikada bir çalışacak şekilde ayarlıyorum."

```yaml
cronnExpression: "*/5 * * * *"
```

### 9.3: extraEmails

Kullanıcıya sor:
"Model sonuçlarının kimlere e-posta ile gitmesini istiyorsun?
E-posta adreslerini paylaşır mısın? (Birden fazlaysa virgülle ayır.)"

Gelen e-posta adreslerini `extraEmails` alanına yaz.

### 9.4: inputDB

Kullanıcıya **sırayla** sor:
1. "Skorlama sorgularının çalışacağı **sunucu adını** paylaşır mısın?
   (Sadece sunucu adını yaz, domain kısmını ben ekleyeceğim.)"
2. "**Database** adını paylaşır mısın?"
3. "**Tablo** adını paylaşır mısın?"
4. "**Schema** adını paylaşır mısın? (Örnek: dbo)"

Sunucu adının sonuna `.deniz.denizbank.com` ekle (zaten varsa ekleme).

`inputDB` alanını bu bilgilerle doldur:
```yaml
inputDB:
  server: "<SUNUCU>.deniz.denizbank.com"
  database: "<DATABASE>"
  table: "<TABLO>"
  schema: "<SCHEMA>"
```

### 9.5: outputDB (Test Ortamı)

Test ortamı için otomatik olarak şu değerleri koy:
```yaml
outputDB:
  server: "SDRCCONSQLD1V02.deniz.denizbank.com"
  database: "StratejikAnalitik"
  table: "ILKERGUR_CRMB_DS_TENDENCY"
  schema: "dbo"
```

Kullanıcıya bilgi ver:
"Test ortamı için outputDB'yi otomatik olarak ayarladım:
- Server: SDRCCONSQLD1V02.deniz.denizbank.com
- Database: StratejikAnalitik
- Table: ILKERGUR_CRMB_DS_TENDENCY
- Schema: dbo"

### values.yaml Değişikliklerini Kaydet

Tüm değişiklikleri kullanıcıya göster ve onayla.

Onay alındıktan sonra ImageId için commit'i güncelle:
```bash
git add Projects/MainProject/values.yaml
git commit --amend --no-edit
```

---

## Adım 10: Test İçin Push

```bash
git push origin <BRANCH_ADI>
```

Kullanıcıya bilgi ver:
"Push tamamlandı. Şimdi OpenShift üzerinde test pipeline'ını kontrol et.
Cron 5 dakikada bir çalışacak şekilde ayarlandı.
- Test başarılı olduysa → bana haber ver, prod ortamına geçelim.
- Hata aldıysan → hata mesajını buraya yapıştır, birlikte çözelim."

Eğer kullanıcı hata paylaşırsa:
1. Hatayı analiz et ve çözüm öner.
2. Gerekli düzeltmeyi yap.
3. **Batch klasöründe değişiklik yaptıysan** commit ID'yi tekrar al
   ve `values.yaml`'daki `ImageId`'yi güncelle.
4. Tekrar commit ve push et.
5. Kullanıcıdan test'i tekrar çalıştırmasını iste.

---

## Adım 11: Prod Ortamına Geçiş Uyarısı

Kullanıcıdan testin başarılı olduğu bilgisini aldıktan sonra şu uyarıyı ver:

"Tebrikler, test başarılı! Şimdi **prod ortamına** geçiyoruz.
⚠️ **Önemli:** Prod ortamında her şeyin hatasız olması gerekiyor.
Özellikle müşteri çoklaması gibi durumların olmaması lazım.
Devam etmek istediğini onaylar mısın?"

Onay bekle, sonra Adım 12'ye geç.

---

## Adım 12: main.py Prod Güncellemesi

`Batch/main.py` dosyasını oku.

Dosya içinde `BIREYSEL_CRM.DBO.ILKERGUR_GNL_DS_MODEL_STATUS_LOG` ifadesini bul
ve `dw_datascientist_stg.[GNL].[DS_MODEL_STATUS_LOG]` ile değiştir.

**Tüm geçen yerlerde** bu değişikliği yap (birden fazla satırda olabilir).

Değişikliği kullanıcıya göster:
"main.py'deki status log tablosu prod ortamına göre güncellendi:
- Eski: BIREYSEL_CRM.DBO.ILKERGUR_GNL_DS_MODEL_STATUS_LOG
- Yeni: dw_datascientist_stg.[GNL].[DS_MODEL_STATUS_LOG]
Uygun mu?"

---

## Adım 13: values.yaml Prod Konfigürasyonu

### 13.1: cronnExpression (Prod)

Kullanıcıya sor:
"Prod ortamında model ayın hangi günü, hangi saatte çalışsın?
(Örnek: Her ayın 1'inde sabah 8'de)

⚠️ **Not:** Sistem UTC+0'da çalışıyor, Türkiye saatinden 3 saat geride.
Yani Türkiye'de sabah 8'de çalışmasını istiyorsan ben otomatik olarak
05:00 UTC'ye ayarlayacağım."

Kullanıcıdan gelen Türkiye saatinden **3 saat çıkar** ve cron expression oluştur.

Örnekler:
- "Her ayın 1'inde sabah 8'de" → `"0 5 1 * *"`
- "Her gün sabah 6'da" → `"0 3 * * *"`
- "Her pazartesi saat 9'da" → `"0 6 * * 1"`

### 13.2: outputDB (Prod Ortamı)

Kullanıcıya sor:
"Prod ortamı için çıktı tablosunun **schema** adını paylaşır mısın?"

Schema alındıktan sonra outputDB'yi güncelle:
```yaml
outputDB:
  server: "S0134DWHDB02.deniz.denizbank.com"
  database: "dw_datascientist_stg"
  table: "DS_TENDENCY_MODEL"
  schema: "<KULLANICIDAN_ALINAN_SCHEMA>"
```

Kullanıcıya bilgi ver:
"outputDB prod ortamına göre güncellendi:
- Server: S0134DWHDB02.deniz.denizbank.com
- Database: dw_datascientist_stg
- Table: DS_TENDENCY_MODEL
- Schema: <KULLANICIDAN_ALINAN_SCHEMA>"

---

## Adım 14: ImageId Güncelle, Son Onay ve Push

### 14.1: Batch Değişikliklerini Commit Et ve ImageId Güncelle

```bash
git add Batch/ Projects/MainProject/values.yaml
git commit -m "Prod ortamı konfigürasyonu tamamlandı"
```

Batch klasörünün son commit ID'sini al:
```bash
git log -1 --format="%H" -- Batch/
```

Bu commit ID'yi `values.yaml`'daki `ImageId` alanına yaz.
Sonra commit'i güncelle:
```bash
git add Projects/MainProject/values.yaml
git commit --amend --no-edit
```

### 14.2: Son Onay

Kullanıcıya tüm değişikliklerin özetini göster ve son onay iste:

"Prod ortamı için tüm ayarlar tamamlandı. Push'lamadan önce özet:

**config.py:**
- MODEL_ID: <değer>
- PRODUCT_CODE: <değer>
- REASON1: <değer>
- CALIBRATION: <değer>

**main.py:**
- SQL dosyaları: <dosya listesi>
- Status log tablosu: dw_datascientist_stg.[GNL].[DS_MODEL_STATUS_LOG]

**values.yaml:**
- cronnExpression: <değer>
- outputDB: S0134DWHDB02 / dw_datascientist_stg / DS_TENDENCY_MODEL
- ImageId: <commit_id>
- extraEmails: <değer>

Her şey doğru mu? Onaylarsan push ediyorum."

### 14.3: Push

Onay alındıktan sonra:
```bash
git push origin <BRANCH_ADI>
```

---

## Adım 15: Skorlama Tamamlandı

"Skorlama/batch predict pipeline kurulumu tamamlandı! İşte yapılanların özeti:

1. ✅ Remote senkronizasyonu yapıldı (git pull --rebase)
2. ✅ config.py — model bilgileri ayarlandı
3. ✅ batch_predict.py — skorlama kodu notebook'tan aktarıldı
4. ✅ SQL sorguları Queries klasörüne eklendi
5. ✅ main.py — SQL çalıştırma alanı ve tablo isimleri güncellendi
6. ✅ requirements.txt — gerekli kütüphaneler eklendi
7. ✅ values.yaml — test ortamında doğrulandı
8. ✅ Prod ortamı konfigürasyonu tamamlandı
9. ✅ ImageId güncel commit ID ile ayarlandı
10. ✅ Son push yapıldı

Başka bir adım veya farklı bir pipeline kurulumu için hazırım."

---

## Hata Yönetimi Protokolü

Bu skill'in herhangi bir adımında kullanıcı hata paylaşırsa:

1. **Hatayı oku ve kategorize et:**
   - Bağlantı hatası (sunucu/veritabanı erişimi)
   - Python hatası (import, syntax, runtime)
   - Git hatası (authentication, conflict, push)
   - OpenShift/CI hatası (build, deploy, resource)
   - Veri hatası (müşteri çoklaması, boş tablo, tip uyumsuzluğu)

2. **Çözüm öner ve uygula:**
   - Dosya düzenlenmesi gerekiyorsa düzenle — ama her düzenlemede
     referans şablonu tekrar oku ve korunan bölümlere dokunmadığını doğrula.
   - Komut çalıştırılması gerekiyorsa komutu ver.
   - Kullanıcının manuel yapması gereken bir şey varsa net talimat ver.
   - **Batch klasöründe değişiklik yaptıysan commit ID'yi tekrar al
     ve values.yaml'daki ImageId'yi güncelle.**

3. **Kaldığın yere dön:**
   - "Hata çözüldü. Adım X'e geri dönüyoruz. [eksik bilgiyi tekrar iste]"
