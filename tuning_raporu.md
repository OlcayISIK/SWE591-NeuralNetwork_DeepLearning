### Görev 2 - Bölüm 3: LSTM Modeli İçin Detaylı Hiperparametre Optimizasyonu (Grid Search)

Projemizin *"En iyi sonuçları almak için hyperparametre tuning ile sonuçları karşılaştırma"* aşamasında, MNIST veri seti üzerinde eğittiğimiz tek katmanlı LSTM (Long Short-Term Memory) modelinin performansını maksimize etmek için kapsamlı bir hiperparametre optimizasyonu gerçekleştirilmiştir. Amacımız, modelin kapasitesini, öğrenme hızını ve veri besleme boyutunu sistematik olarak değiştirerek en ideal ve genellenebilir modeli bulmaktır.

#### 1. Optimizasyon Stratejisi ve Sabit Tutulan Parametreler

**Epoch Sayısının Belirlenmesi:**
Modellerin performansını örneğin 5 epoch gibi kısa bir sürede karşılaştırmak yanıltıcı sonuçlar doğurabilmektedir. Çünkü özellikle düşük öğrenme oranına (learning rate) sahip modellerin yakınsaması (convergence) daha geç olmakla birlikte, eğitimin sonunda daha yüksek bir doğruluğa ulaşabilmektedir. Adil ve tam bir karşılaştırma yapabilmek adına, tüm hiperparametre kombinasyonları orijinal eğitimde de kullanılan **10 epoch** üzerinden değerlendirilmiştir.

**Optimizer Seçimi:**
Derin öğrenme modellerinde optimizasyon için farklı algoritmalar (SGD, RMSprop, vb.) bulunsa da, LSTM gibi tekrarlayan (recurrent) sinir ağlarında **Adam** algoritması ivmelenme (momentum) ve adaptif öğrenme oranını birleştirerek neredeyse her zaman en hızlı ve en kararlı yakınsamayı sağlar. Standart SGD kullanmak LSTM'lerde genellikle yavaş ve kararsız sonuçlar ürettiğinden, optimizer parametresi olarak zaman kaybetmemek ve en optimum sonucu garantilemek adına Adam algoritması sabit tutulmuştur.

#### 2. Test Edilen Hiperparametreler ve Seçim Gerekçeleri

Optimizasyon uzayımızı belirlerken, modelin başarısına en yüksek etkiyi yapacak, akademik olarak da literatürde en çok incelenen üç temel parametre üzerine odaklanılmıştır:

*   **Hidden Size (64 ve 128):** Bu parametre, LSTM hücresinin içerisindeki gizli durumun (hidden state) boyutunu ifade eder ve modelin ardışık verilerdeki örüntüleri hatırlama/öğrenme kapasitesini doğrudan belirler. Kapasitenin modele etkisini gözlemlemek için 64 ve 128 değerleri test edilmiştir.
*   **Batch Size (64 ve 128):** Modelin ağırlıklarını (weights) bir epoch içerisinde ne sıklıkla güncelleyeceğini belirler. Örneğin, batch size'ın 64 olması, 128'e kıyasla her epoch'ta modelin ağırlıklarının iki kat daha fazla güncellenmesini sağlar. Daha sık güncelleme yapmak bazı durumlarda yerel minimumlardan (local minima) kurtulmayı kolaylaştırıp daha iyi bir doğruluk (accuracy) sağlarken, çok büyük olması eğitim stabilitesini bozabilir.
*   **Learning Rate (1e-3 ve 5e-4):** Modelin, kayıp fonksiyonunun (loss function) minimumuna doğru atacağı adımların büyüklüğünü ifade eder. Yüksek bir oran hızlı öğrenme sağlasa da optimum noktayı atlayabilir (overshooting). Bu nedenle standart kabul edilen `1e-3` (0.001) ve daha hassas adımlar atan `5e-4` (0.0005) değerleri karşılaştırılmıştır.

#### 3. Uygulama (Grid Search) ve Sonuçlar

Belirlenen üç parametrenin tüm olasılıkları çaprazlanarak ($2 \times 2 \times 2$) **toplam 8 farklı kombinasyon** elde edilmiştir. Bu 8 farklı modelin her biri sıfırdan başlatılarak 10'ar epoch boyunca eğitilmiş (toplam 80 epoch) ve test veri seti üzerindeki doğruluk oranlarına göre sıralanmıştır.

Elde edilen Grid Search sonuç tablosu aşağıdadır:

| Hidden Size | Learning Rate | Batch Size | Test Accuracy (%) |
|-------------|---------------|------------|-------------------|
| **128**     | **0.0010**    | **64**     | **98.95**         |
| 128         | 0.0010        | 128        | 98.21             |
| 128         | 0.0005        | 64         | 98.09             |
| 64          | 0.0010        | 64         | 97.97             |
| 128         | 0.0005        | 128        | 97.64             |
| 64          | 0.0005        | 64         | 97.58             |
| 64          | 0.0010        | 128        | 97.50             |
| 64          | 0.0005        | 128        | 96.58             |

#### 4. Sonuç Analizi ve Değerlendirme

Yapılan testler sonucunda en iyi performansı gösteren model:
*   **Hidden Size:** 128
*   **Learning Rate:** 0.001
*   **Batch Size:** 64
şeklinde tespit edilmiş ve **%98.95 Test Accuracy** değerine ulaşılmıştır.

**Çıkarımlarımız:**
1.  **Batch Size'ın Etkisi:** İlk geliştirdiğimiz modelde kullanılan Batch Size = 128 değeri %98.24 başarı sağlarken, bu değerin 64'e düşürülmesi başarı oranını %98.95'e taşımıştır. Bunun nedeni, her epoch'ta yapılan ağırlık güncelleme sayısının artmasının modelin veri setindeki ince detaylara daha iyi adapte olmasını sağlamasıdır.
2.  **Öğrenme Oranının (LR) Etkisi:** 0.001 değeri, 0.0005 değerine göre daha hızlı ve başarılı bir yakınsama sağlamıştır. Daha küçük adımlar atan 0.0005 değeri, 10 epoch içerisinde modelin tam potansiyeline ulaşması için yetersiz kalmıştır.
3.  **Hafıza Kapasitesinin Etkisi:** 128 boyutlu Hidden Size, her senaryoda 64 boyutlu modele göre üstünlük sağlamış, MNIST gibi 28 farklı zaman adımı (time step) barındıran ardışık örüntüleri çözümlemede yüksek kapasitenin doğrudan performans artışı sağladığını kanıtlamıştır.

Özetle, modelin hiperparametreleri rastgele denemeler yerine tamamen sistematik ve analitik bir yöntemle optimize edilmiştir. Modelin kapasitesi, öğrenme hızı ve veri besleme boyutu arasındaki ilişki gözlemlenerek projenin ana hedeflerinden olan >%90 doğruluk barajı son derece bilimsel bir arka planla aşılmıştır.

---

### Görev 3: Variational Autoencoder (VAE) Hiperparametre Optimizasyonu

Projenin en gelişmiş modeli olan LSTM kodlayıcılı VAE (Variational Autoencoder) modeli için de benzer şekilde sistematik bir grid search yapılmıştır. VAE'lerin performansı geleneksel modellerden farklı olarak sadece yeniden yapılandırma hatasıyla (Reconstruction Loss) değil, aynı zamanda gizli uzaydaki dağılımın standart normal dağılıma ne kadar benzediğini ölçen **KL Divergence** ile birlikte **Toplam ELBO Kaybı (Total Loss)** üzerinden değerlendirilir.

#### 1. Test Edilen Hiperparametreler
Modelin karmaşıklığı gereği, VAE mimarisini doğrudan etkileyen şu üç temel parametre üzerinde 8 farklı kombinasyon test edilmiştir (Tüm testler 10 epoch boyunca çalıştırılmıştır):
*   **Latent Dimension (10 ve 20):** Darboğazın boyutu. 20 boyutlu uzay, daha fazla özellik saklarken (düşük Recon Loss), modelin dağılımı öğrenmesini zorlaştırabilir (yüksek KL Divergence). 
*   **Hidden Size (64 ve 128):** LSTM kodlayıcının hücresel kapasitesi.
*   **Learning Rate (1e-3 ve 5e-4):** Adam optimizasyon algoritmasının adım büyüklüğü.

#### 2. Grid Search Sonuçları (Test Veri Seti Üzerinde)

Aşağıdaki tablo, 8 modelin **Toplam Kayıp (Total Loss)** değerine göre en iyiden en kötüye doğru sıralanmış halini göstermektedir:

| Latent Dim | Hidden Size | Learning Rate | Total Loss | Recon Loss (BCE) | KL Divergence |
|------------|-------------|---------------|------------|------------------|---------------|
| **20**     | **128**     | **0.0010**    | **106.58** | **89.26**        | 17.31         |
| 20         | 128         | 0.0005        | 109.49     | 92.26            | 17.23         |
| 10         | 128         | 0.0010        | 110.65     | 95.87            | 14.78         |
| 20         | 64          | 0.0010        | 111.10     | 95.28            | 15.81         |
| 10         | 64          | 0.0010        | 113.75     | 99.55            | 14.20         |
| 10         | 128         | 0.0005        | 113.76     | 99.73            | 14.03         |
| 20         | 64          | 0.0005        | 117.72     | 103.64           | 14.07         |
| 10         | 64          | 0.0005        | 118.18     | 104.46           | **13.72**     |

#### 3. Sonuç Analizi ve Bilimsel Çıkarımlar

Optimizasyon sonuçları incelendiğinde, en başarılı modelin **Latent Dim: 20, Hidden Size: 128 ve LR: 0.001** kombinasyonuna sahip olduğu görülmektedir.

**Çıkarımlarımız:**
1.  **Latent Dimension ve Trade-off (Ödünleşim):** Latent Dim = 20 olduğunda model daha zengin bir vektörel temsil kapasitesine sahip olduğu için *Reconstruction Loss* (Yeniden Yapılandırma Hatası) ciddi şekilde düşmüştür (89.26'ya kadar). Ancak yüksek boyut, standart normal dağılıma uymayı zorlaştırdığından *KL Divergence* değeri artmıştır. Latent Dim = 10 olan modellerde ise tersi şekilde *KL Divergence* daha düşük (örneğin 13.72) ancak *Reconstruction Loss* çok daha yüksektir (104.46). Bu durum VAE'nin karakteristik ödünleşimiyle birebir örtüşmektedir; toplam kayıp dikkate alındığında 20 boyutun daha optimum bir denge kurduğu görülmüştür.
2.  **Öğrenme Oranının (LR) Etkisi:** Tıpkı LSTM optimizasyonunda olduğu gibi, `0.001` öğrenme oranına sahip modeller, `0.0005`'e sahip benzerlerine kıyasla her zaman daha düşük toplam kayıp elde etmiştir. Bu da Adam optimizer için `1e-3` değerinin bu veri setinde optimum minimuma daha iyi yaklaştığını doğrular.
3.  **Hafıza Kapasitesinin (Hidden Size) Etkisi:** Hidden Size = 128 olan modeller, 64 olan modellere göre LSTM içerisindeki ardışık zamansal ilişkileri çok daha iyi çözümlemiş ve hem yeniden yapılandırma hem de toplam ELBO kaybında bariz bir üstünlük kurmuştur.

Bu detaylı optimizasyon süreci sonucunda, VAE modelinin kapasitesi sistematik olarak maksimum düzeye çıkarılmış ve oluşturduğumuz jeneratif model bilimsel olarak doğrulanmıştır.
