---
name: ci-setup
description: >
  ML projeleri için CI pipeline kurulumunu adım adım yürütür.
  get_data.py, train.py ve values.yaml dosyalarını doldurur,
  git branch oluşturur ve CI sürecini sonuna kadar takip eder.
  Bu skill şu durumlarda tetiklenir: CI kurulumu, CI başlatma,
  pipeline hazırlama, model deploy, proje şablonu doldurma,
  "CI'ı kur", "projeyi hazırla", "deploy için hazırla" gibi ifadeler.
---

# CI Pipeline Kurulum Skill'i

Bu skill, ML projelerinin CI pipeline'ını baştan sona kurar.
Kullanıcıya her adımı sırayla sor, bir adım tamamlanmadan diğerine geçme.

## Genel Kurallar

- Her adımı sırayla takip et, atlama yapma.
- Kullanıcıdan bilgi beklerken, kullanıcı bir **hata** paylaşırsa:
  1. Önce hatayı analiz et ve çözüm öner.
  2. Hata çözüldükten sonra kaldığın adıma geri dön ve eksik bilgiyi tekrar iste.
  3. Kullanıcıya hangi adımda olduğunu hatırlat.
- Dosya düzenlerken şablon yapısını **kesinlikle bozma**. Şablon yapısını anlamak
  için bu skill klasöründeki referans dosyaları kullan (aşağıda detaylandırılmıştır).
- Her dosya değişikliğinden önce mevcut dosyayı oku ve yapıyı anla.
- Türkçe iletişim kur, teknik terimleri olduğu gibi bırak (git, branch, CI, deploy vb.).

## Şablon Referans Dosyaları

Bu skill klasöründe şu referans dosyalar bulunur:

- **`get_data_template.py`** — get_data.py'nin orijinal şablon hali.
  `# === BU BÖLÜME DOKUNMA ===` ile işaretlenmiş bölümlere kesinlikle dokunma.
  Sadece `# === BURAYA YAZ ===` ile işaretlenmiş bölümleri doldur.

- **`train_template.py`** — train.py'nin orijinal şablon hali.
  `# === BU BÖLÜME DOKUNMA ===` ile işaretlenmiş bölümlere kesinlikle dokunma.
  Sadece `# === BURAYA YAZ ===` ile işaretlenmiş bölüme eğitim kodunu yerleştir.
  Model kaydetme bölümünde `dill` kütüphanesi kullanılır — bu yapıyı koru.

- **`values-example.yaml`** — values.yaml'ın örnek yapısı.
  enableCI ve "For Batch Deployment" bölümlerinin tam formatını gösterir.

**Her dosya düzenlemesinden önce** ilgili referans dosyayı oku ve mevcut dosyayla
karşılaştır. Düzenleme yaparken referanstaki yapıyı baz al.

---

## Adım 1: Proje Reposunu Klonla

Kullanıcıya sor:
- "Proje reposunun git URL'ini paylaşır mısın?"

URL alındıktan sonra:
```bash
git clone <REPO_URL>
cd <REPO_ADI>
```

Klonlama başarılı olduktan sonra repo yapısını incele:
```bash
find . -type f -name "*.py" -o -name "*.yaml" -o -name "*.yml" -o -name "*.ipynb" | head -40
```

Kullanıcıya repo yapısını özetle ve Adım 2'ye geç.

---

## Adım 2: get_data.py Dosyasını Doldur

1. Önce bu skill klasöründeki `get_data_template.py` referans dosyasını oku.
2. Sonra repodaki `CI/components/get_data.py` dosyasını oku.
3. Referans dosyadaki `# === BU BÖLÜME DOKUNMA ===` bölümlerini tespit et.

Kullanıcıya **sırayla** şunları sor:

1. "Veritabanı sunucu adresini (server) paylaşır mısın? (Sadece sunucu adını yaz,
   domain kısmını ben ekleyeceğim.)"
2. "Veritabanı adını (database) paylaşır mısın?"
3. "Eğitim verisini çekmek için kullanılacak SQL sorgusunu paylaşır mısın?
   (Doğrudan yapıştırabilirsin veya bir .sql dosyası varsa yolunu verebilirsin.)"

### Sunucu Adı Kuralı

Kullanıcının verdiği sunucu adının sonuna `.deniz.denizbank.com` ekle.
Eğer kullanıcı zaten tam domain ile verdiyse (`.deniz.denizbank.com` içeriyorsa)
tekrar ekleme.

Örnek:
- Kullanıcı `SDRCCONSQLD1V02` derse → `SDRCCONSQLD1V02.deniz.denizbank.com`
- Kullanıcı `SDRCCONSQLD1V02.deniz.denizbank.com` derse → olduğu gibi bırak

**Düzenleme kuralları:**
- SQL sorgusunu alırken, eğer kullanıcı bir dosya yolu verdiyse o dosyayı oku.
- Referans şablondaki `# === BU BÖLÜME DOKUNMA ===` blokları arasındaki
  kodu **olduğu gibi** koru.
- Sadece `# === BURAYA YAZ ===` ile işaretlenmiş yerlere değerleri yerleştir.
- Değişiklik yaptıktan sonra dosyayı kullanıcıya göster ve onayla:
  "get_data.py şu şekilde güncellendi, uygun mu?"

Onay alındıktan sonra Adım 3'e geç.

---

## Adım 3: train.py Dosyasını Doldur

### Aşama 3a: Kaynak Notebook'u Bul

Repo içinde `CI - *.ipynb` formatında notebook ara:
```bash
find . -name "CI - *.ipynb" -o -name "CI-*.ipynb" -o -name "ci - *.ipynb" -o -name "ci-*.ipynb" 2>/dev/null
```

**Eğer notebook bulunamazsa**, kullanıcıya doğrudan sor:
"CI notebook'unu bulamadım. Eğitim kodunu içeren notebook veya
Python dosyasının tam yolunu paylaşır mısın?"

Eğer birden fazla notebook bulunursa, listeyi kullanıcıya göster ve
hangisini kullanacağını sor.

### Aşama 3b: Eğitim Kodunu train.py'ye Aktar

1. Önce bu skill klasöründeki `train_template.py` referans dosyasını oku.
2. Sonra repodaki `CI/components/train.py` dosyasını oku.
3. Referans dosyadaki korunan ve düzenlenecek bölümleri tespit et.
4. Notebook'u oku ve eğitim kodunu çıkar.

**Korunan bölümler (kesinlikle değiştirme):**
- `# === BU BÖLÜME DOKUNMA ===` ile başlayan ve
  `# === BU BÖLÜMÜN SONU ===` ile biten her blok.
- Bunlar genellikle data okuma (dosya başı) ve model kaydetme (dosya sonu)
  bölümleridir.
- Model kaydetme bölümünde `dill` ile serialization yapılır — bu kısma dokunma.

**Model kaydetme hakkında önemli not:**
Şablondaki model kaydetme bölümü `dill` kütüphanesi kullanır. Notebook'ta
`pickle`, `joblib` veya başka bir serialization yöntemi kullanılıyorsa,
bunları **dahil etme**. Şablondaki `dill` ile kaydetme mekanizması zaten
modeli doğru şekilde kaydedecektir. Notebook'tan sadece model nesnesinin
oluşturulması ve eğitilmesi (fit) kodunu al, kaydetme kodunu alma.

**Notebook'tan aktarılacak kod:**
- Feature engineering adımları
- Model tanımlama ve eğitim (fit) kodu
- Hyperparameter konfigürasyonu
- Metrik hesaplama

**Notebook'tan aktarılMAyacak kod:**
- EDA (Exploratory Data Analysis) hücreleri
- Görselleştirme kodları (matplotlib, seaborn plotları)
- Deneme-yanılma / debug hücreleri
- `display()`, `print()` ile sadece inceleme amaçlı yazılmış satırlar
- Model kaydetme/serialization kodları (pickle.dump, joblib.dump vb.)
  — şablondaki `dill` mekanizması bunu halledecek

**Yerleştirme:**
- Notebook'tan çıkarılan kodu `# === BURAYA YAZ ===` bloğunun içine koy.
- Notebook'taki import'ları train.py'nin başına ekle (mevcut import'ları silmeden).
  `dill` import'u zaten şablonda var, tekrar ekleme.
- Değişken adları çakışıyorsa notebook'takini uyarla, şablondakini değiştirme.

Düzenleme tamamlandığında dosyayı kullanıcıya göster:
"train.py şu şekilde güncellendi. Korunan bölümlere (data okuma ve model
kaydetme) dokunmadım, model dill ile kaydedilecek. Uygun mu?"

Onay alındıktan sonra Adım 4'e geç.

---

## Adım 4: values.yaml Güncelle, Git Ayarla ve Push Et

### Aşama 4a: values.yaml İlk Düzenleme

1. Önce bu skill klasöründeki `values-example.yaml` referans dosyasını oku.
2. Sonra repodaki `Projects/MainProject/values.yaml` dosyasını oku.
3. Referans dosyayı baz alarak şu değişiklikleri yap:
   - `enableCI` değerini `True` olarak ayarla.
   - "For Batch Deployment" yazan bölümün altındaki yorum satırlarını kaldır
     (satır başındaki `#` karakterlerini sil). Referans dosyadaki yapıyı
     takip ederek doğru satırları uncomment et.

Değişikliği kullanıcıya göster ve onayla.

### Aşama 4b: Git Kullanıcı Bilgilerini Ayarla

Kullanıcıya **sırayla** sor:
1. "Git için kullanıcı adını paylaşır mısın? (Örnek: ilker.yilmaz)"
2. "Git için e-posta adresini paylaşır mısın? (Örnek: ilker.yilmaz@denizbank.com)"

Bilgiler alındıktan sonra terminalde çalıştır:
```bash
git config --global user.name "<KULLANICI_ADI>"
git config --global user.email "<EMAIL>"
```

### Aşama 4c: Yeni Branch, Commit ve Commit ID

Kullanıcıya sor:
- "Yeni branch için bir isim belirleyelim. Önerin var mı?
  (Örnek: `feature/ci-<model-adi>`)"

Branch adı belirlendikten sonra:
```bash
git checkout -b <BRANCH_ADI>
git add CI/components/get_data.py CI/components/train.py Projects/MainProject/values.yaml
git commit -m "CI pipeline konfigürasyonu tamamlandı"
```

**Push'lamadan ÖNCE**, CI klasöründeki son commit ID'yi al ve values.yaml'a yaz:
```bash
git log -1 --format="%H" -- CI/
```

Bu komutun çıktısı commit ID'dir. Bu değeri `Projects/MainProject/values.yaml`
dosyasındaki `experimentId` alanına yaz:
```bash
# values.yaml'daki experimentId'yi güncelle
sed -i 's/experimentId:.*/experimentId: <COMMIT_ID>/' Projects/MainProject/values.yaml
```

Sonra bu değişikliği de commit'e ekle:
```bash
git add Projects/MainProject/values.yaml
git commit --amend --no-edit
```

Şimdi push et:
```bash
git push origin <BRANCH_ADI>
```

Push tamamlandıktan sonra kullanıcıya bilgi ver:
"Push tamamlandı. Şimdi GitHub/GitLab üzerinden bir **Pull Request**
oluşturman gerekiyor. PR'ı oluşturduktan ve merge edildikten sonra
bana haber ver, bir sonraki adıma geçelim."

---

## Adım 5: CI Durumunu Takip Et

Kullanıcıya bilgi ver:
"PR merge edildikten sonra CI, Red Hat OpenShift üzerinde otomatik
olarak tetiklenecek. OpenShift'teki pipeline'ı kontrol et.
- Başarılı olduysa → bana haber ver, Adım 6'ya geçelim.
- Hata aldıysan → hata mesajını buraya yapıştır, birlikte çözelim."

Eğer kullanıcı bir hata paylaşırsa:
1. Hatayı analiz et (yaygın sorunlar: import eksiklikleri, veri tipi
   uyumsuzlukları, bellek hataları, bağlantı sorunları).
2. Gerekli düzeltmeyi yap (get_data.py, train.py veya values.yaml'da).
   Düzenleme yaparken **korunan bölümleri kontrol et**, onlara dokunma.
3. Düzeltilmiş dosyayı tekrar commit ve push et.
4. Kullanıcıdan CI'ı tekrar çalıştırmasını iste.
5. CI başarılı olana kadar bu döngüyü tekrarla.

---

## Adım 6: OpenShift Experiment ID'yi values.yaml'a Yaz

CI başarıyla tamamlandığında kullanıcıya sor:
"CI başarılı oldu, tebrikler! Şimdi OpenShift üzerindeki
**Experiment ID**'yi paylaşır mısın?"

Experiment ID alındıktan sonra:
1. `Projects/MainProject/values.yaml` dosyasını aç.
2. `experimentId` alanını OpenShift'ten alınan yeni ID ile güncelle.
   (Bu, Adım 4c'de yazdığımız commit ID'nin yerini alacak.)
3. Değişikliği kaydet, commit ve push et:

```bash
git add Projects/MainProject/values.yaml
git commit -m "Experiment ID güncellendi (OpenShift): <EXPERIMENT_ID>"
git push origin <BRANCH_ADI>
```

---

## Adım 7: Remote Senkronizasyonu ve CI Tamamlama

Sonraki adıma (skorlama/batch predict) geçmeden önce remote'taki
değişiklikleri al:
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

Rebase başarılı olduktan sonra kullanıcıya özet sun:

"CI pipeline kurulumu tamamlandı! İşte yapılanların özeti:

1. ✅ Proje reposu klonlandı
2. ✅ get_data.py — sunucu (.deniz.denizbank.com), veritabanı ve SQL sorgusu ayarlandı
3. ✅ train.py — eğitim kodu notebook'tan aktarıldı (model dill ile kaydediliyor)
4. ✅ values.yaml — CI aktifleştirildi, batch deployment açıldı
5. ✅ Git kullanıcı bilgileri ayarlandı, branch oluşturuldu ve push edildi
6. ✅ CI, OpenShift üzerinde başarıyla çalıştı
7. ✅ OpenShift Experiment ID values.yaml'a yazıldı
8. ✅ Remote senkronizasyonu yapıldı (git pull --rebase)

Bir sonraki adım olarak skorlama/batch predict kurulumuna
geçmek istersen hazırım."

---

## Hata Yönetimi Protokolü

Bu skill'in herhangi bir adımında kullanıcı hata paylaşırsa:

1. **Hatayı oku ve kategorize et:**
   - Bağlantı hatası (sunucu/veritabanı erişimi)
   - Python hatası (import, syntax, runtime)
   - Git hatası (authentication, conflict, push)
   - OpenShift/CI hatası (build, deploy, resource)

2. **Çözüm öner ve uygula:**
   - Dosya düzenlenmesi gerekiyorsa düzenle — ama her düzenlemede
     referans şablonu tekrar oku ve korunan bölümlere dokunmadığını doğrula.
   - Komut çalıştırılması gerekiyorsa komutu ver.
   - Kullanıcının manuel yapması gereken bir şey varsa net talimat ver.

3. **Kaldığın yere dön:**
   - "Hata çözüldü. Adım X'e geri dönüyoruz. [eksik bilgiyi tekrar iste]"
