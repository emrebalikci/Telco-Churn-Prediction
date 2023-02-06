#Churn, telekomünikasyon şirketleri gibi abonelik tabanlı hizmetler için önemli bir iş metriğidir.
#Bu proje, IBM örnek veri kümelerinden indirilen verileri kullanan bir kayıp analizi gösterir.
#Müşteri kaybıyla ilişkili değişkenleri belirlemek için R istatistiksel programlama dilini kullanacağız.

##### Projede kullanılacak olan gerekli kütüphanelerin yuklenmesi
library(plyr)
library(randomForest)
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(gridExtra)

# kullanılacak verisetinin sisteme yüklenmesi.
veri <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

#glimpse() fonksiyonu veri setimizdeki verilerin yapsını ve iceriğini incelemek icin kullanılır.
glimpse(veri)

# customerID: Müşteri Kimliği
# genderCustomer: cinsiyet (kadın, erkek)
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# PartnerWhether: müşterinin bir ortağı var mı yok mu (Evet, Hayır)
# Dependents: Müşterinin bağımlıları olup olmadığı (Evet, Hayır)
# görev süresi: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin online güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik desteği olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturalandırması olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar

##### VERİ ÖN İŞLEME ####

#Eksik veri olup olmadığını görerek başlayalım.
sapply(veri, function(x) sum(is.na(x)))

#Toplam Masraflar değişkeninde eksik değerleri olan 11 durum vardır. Bu özel durumları görelim.
veri[is.na(veri$TotalCharges),]

#Churn değişkeninin incelenmesi, bunların hepsinin hala abone olan müşteriler olduğunu gösteriyor. Örneklemimizin yüzde kaçı eksik değerlere sahip bu alt kümedir?
sum(is.na(veri$TotalCharges))/nrow(veri)

#Bu alt küme, verilerimizin %0,16'sıdır ve oldukça küçüktür. Daha sonraki analizlerimize uyum sağlamak için bu vakaları kaldıracağız. Bu temizlenmiş veriye temiz_veri diyelim.
temiz_veri <- veri[complete.cases(veri), ]

#SeniorCitizen değişkeni evet/hayır yerine '0/1' olarak kodlanmıştır. Daha sonraki grafikleri ve modelleri yorumlamamızı kolaylaştırmak için bunu yeniden kodlayabiliriz.

temiz_veri$SeniorCitizen <- as.factor(mapvalues(temiz_veri$SeniorCitizen,
                                                 from=c("0","1"),
                                                 to=c("No", "Yes")))

#MultipleLines de ğişkeni, PhoneService değişkenine bağlıdır; burada ikinci değişken için bir "hayır", önceki değişken için otomatik olarak "hayır" anlamına gelir. MultipleLines değişkeni için "Hayır"a "Telefon hizmeti yok" yanıtını yeniden kodlayarak grafiklerimizi ve modellememizi daha da kolaylaştırabiliriz.
temiz_veri$MultipleLines <- as.factor(mapvalues(temiz_veri$MultipleLines, 
                                                 from=c("No phone service"),
                                                 to=c("No")))


# Benzer şekilde, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV ve StreamingMovies değişkenlerinin tümü OnlineService değişkenine bağlıdır. Bu değişkenler için "İnternet hizmeti yok" ile "Hayır" arasındaki yanıtları yeniden kodlayacağız.
for(i in 10:15){
  temiz_veri[,i] <- as.factor(mapvalues(temiz_veri[,i],
                                         from= c("No internet service"), to= c("No")))
}


# Grafikler veya modelleme için müşteri kimliği değişkenine ihtiyacımız olmayacak, bu nedenle kaldırılabilir.

temiz_veri$customerID <- NULL


##Çeşitli modelleme tekniklerimizin performansını değerlendirmek için verileri eğitim ve test alt kümelerine ayırabiliriz. 
# Eğitim verilerini modelleyeceğiz ve bu model parametrelerini test verileriyle tahminler yapmak için kullanacağız. 
# Bu veri alt kümelerine dtrain ve dtest diyelim.
# bu alt kümeleri oluşturmak için tüm numuneden rastgele numune alacağız. Örnekleme için kullanılan rasgele sayı üretecini sıfırlamak için 'set.seed()' işlev argümanı değiştirilebilir. 
# Eğitim alt kümesi, orijinal numunenin kabaca %70'i olacak ve kalan kısım test alt kümesi olacaktır.



set.seed(56)
split_train_test <- createDataPartition(temiz_veri$Churn,p=0.7,list=FALSE)
dtrain<- temiz_veri[split_train_test,]
dtest<- temiz_veri[-split_train_test,]





#### Tanımlayıcı İstatistikler İçin Veri Görselleştirme ####
#Verilerinizi modellemeye başlamadan önce, bazı basit grafiklerde verilerinizin tanımlayıcı istatistiklerini inceleyelim.


#Gender plot
p1 <- ggplot(temiz_veri, aes(x = gender)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Senior citizen plot
p2 <- ggplot(temiz_veri, aes(x = SeniorCitizen)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Partner plot
p3 <- ggplot(temiz_veri, aes(x = Partner)) + geom_bar() + geom_text(aes(y = ..count.. -200,   label = paste0(round(prop.table(..count..),4) * 100, '%')), stat = 'count', position = position_dodge(.1), size = 3)

#Dependents plot
p4 <- ggplot(temiz_veri, aes(x = Dependents)) + geom_bar() + geom_text(aes(y = ..count.. -200, label = paste0(round(prop.table(..count..),4) * 100, '%')),  stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Plot demographic data within a grid
grid.arrange(p1, p2, p3, p4, ncol=2)



#Bu demografik grafiklerden, örneğin cinsiyet ve partner durumu arasında eşit olarak bölündüğünü fark ettik. Örneklemin bir azınlığı yaşlı vatandaşlardır ve bir azınlığın bağımlıları vardır.

#Sunulan çeşitli hizmetler aşağıda gösterilmiştir.


#Phone service plot
p5 <- ggplot(temiz_veri, aes(x = PhoneService)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Multiple phone lines plot
p6 <- ggplot(temiz_veri, aes(x = MultipleLines)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Internet service plot
p7 <- ggplot(temiz_veri, aes(x = InternetService)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Online security service plot
p8 <- ggplot(temiz_veri, aes(x = OnlineSecurity)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Online backup service plot
p9 <- ggplot(temiz_veri, aes(x = OnlineBackup)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Device Protection service plot
p10 <- ggplot(temiz_veri, aes(x = DeviceProtection)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Tech Support service plot
p11 <- ggplot(temiz_veri, aes(x = TechSupport)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Streaming TV service plot
p12 <- ggplot(temiz_veri, aes(x = StreamingTV)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Streaming Movies service plot
p13 <- ggplot(temiz_veri, aes(x = StreamingMovies)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Plot service data within a grid
grid.arrange(p5, p6, p7,
             p8, p9, p10,
             p11, p12, p13,
             ncol=3)


##Örneğin çoğu, tek bir telefon hattıyla telefon hizmetine sahiptir. Fiber optik internet bağlantısı, DSL internet hizmetinden daha popülerdir ve her çevrimiçi hizmetin az sayıda kullanıcısı vardır.
#Kalan kategorik değişkenler sözleşme ve ödeme durumu ile ilgilidir.



p14 <- ggplot(temiz_veri, aes(x = Contract)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Paperless billing plot
p15 <- ggplot(temiz_veri, aes(x = PaperlessBilling)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Payment method plot
p16 <- ggplot(temiz_veri, aes(x = PaymentMethod)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)

#Plot contract data within a grid
grid.arrange(p14, p15, p16, ncol=1)


##Örneklemin yaklaşık yarısı, bir ve iki yıllık sözleşmeler arasında bölünmüş, aydan aya sözleşmelerdedir. Numunenin çoğu kağıtsız faturada ve elektronik çekle ödeme yapıyor.

#Nicel değişkenlerin dağılımlarına bakalım.


#Tenure histogram
p17 <- ggplot(temiz_veri, aes(x = tenure)) +
  geom_histogram(binwidth = 1) +
  labs(x = "Months",
       title = "Tenure Distribtion")

#Monthly charges histogram
p18 <- ggplot(temiz_veri, aes(x = MonthlyCharges)) +
  geom_histogram(binwidth = 5) +
  labs(x = "Dollars (binwidth = 5)",
       title = "Monthly charges Distribtion")

#Total charges histogram
p19 <- ggplot(temiz_veri, aes(x = TotalCharges)) +
  geom_histogram(binwidth = 100) +
  labs(x = "Dollars (binwidth = 100)",
       title = "Total charges Distribtion")

#Plot quantitative data within a grid
grid.arrange(p17, p18, p19, ncol=1)



#Görev süresi değişkeni kuyruklarda istiflenir, bu nedenle müşterilerin büyük bir oranı ya en kısa (1 ay) ya da en uzun (72 ay) görev süresine sahiptir. MonthlyCharges değişkeni, en düşük oranların yakınında büyük bir yığınla, kabaca normal olarak ayda yaklaşık 80 ABD doları civarında dağıtılıyor gibi görünüyor. TotalCharges değişkeni, düşük miktarlara yakın büyük bir yığınla pozitif olarak çarpıktır.

#Son olarak, ana sonuç değişkenimiz olan churn'u inceleyelim.



p20 <- ggplot(temiz_veri, aes(x = Churn)) +
  geom_bar() +
  geom_text(aes(y = ..count.. -200, 
                label = paste0(round(prop.table(..count..),4) * 100, '%')), 
            stat = 'count', 
            position = position_dodge(.1), 
            size = 3)
p20



#Numunemizin kabaca dörtte biri artık müşteri değil. Bazı sınıflandırma modelleme teknikleriyle çalkantıları tahmin etmeye çalışalım.



#####4. İstatistiksel modelleme
#Eğitim verisi alt kümesinde üç modelleme tekniği uygulanacaktır. Bu tekniklerden model parametreleri, test alt kümesi üzerinde tahminler yapacaktır. Doğruluğu, hem doğru tahminlerin yüzdesi hem de karışıklık matrisleri açısından inceleyeceğiz.

#Karar ağacı analizi
#Karar ağacı analizi, ağaç benzeri karar modellerini ve bunların olası sonuçlarını kullanan bir sınıflandırma yöntemidir. Bu yöntem, makine öğrenmesi analizinde en sık kullanılan araçlardan biridir. Karar ağaçları için özyinelemeli bölümleme yöntemlerini kullanmak için rpart kitaplığını kullanacağız. Bu keşif yöntemi, hiyerarşik bir formatta kayıpla ilgili en önemli değişkenleri tanımlayacaktır.


tr_fit <- rpart(Churn ~., data = dtrain, method="class")
rpart.plot(tr_fit)

# Bu karar ağacından aşağıdakileri yorumlayabiliriz:
#   
#   Sözleşme değişkeni en önemlisidir. Aydan aya sözleşmeleri olan müşterilerin ayrılma olasılığı daha yüksektir.
# DSL internet hizmetine sahip müşterilerin ayrılma olasılığı daha düşüktür.
# 15 aydan daha uzun süre kalan müşterilerin ayrılma olasılığı daha düşüktür.
# Şimdi, test alt kümesindeki dalgalanmayı ne kadar iyi tahmin ettiğini araştırarak karar ağacı modelinin tahmin doğruluğunu değerlendirelim. Sınıflandırma doğruluğunun kullanışlı bir göstergesi olan karışıklık matrisi ile başlayacağız. Aşağıdaki bilgileri görüntüler:
#   
#   gerçek pozitifler (TP): Bunlar, evet tahmin ettiğimiz (çalıştılar) ve çalkaladıkları durumlardır.
# gerçek negatifler (TN): Hayır olduğunu tahmin ettik ve onlar da değişmedi.
# yanlış pozitifler (FP): Evet olduğunu tahmin ettik, ancak gerçekte çalkalamadılar. ("Tip I hatası" olarak da bilinir)
# yanlış negatifler (FN): Hayır tahmininde bulunduk, ancak gerçekte çalkalandılar. ("Tip II hatası" olarak da bilinir)
# Karar ağacı modelimiz için karışıklık matrisini inceleyelim.



tr_prob1 <- predict(tr_fit, dtest)
tr_pred1 <- ifelse(tr_prob1[,2] > 0.5,"Yes","No")
table(Predicted = tr_pred1, Actual = dtest$Churn)

# 
# Çapraz girişler, sol üst kısım TN ve sağ alt kısım TP olacak şekilde doğru tahminlerimizi verir. Sağ üst kısım FN'yi verirken sol alt kısım FP'yi verir. Bu karışıklık matrisinden, modelin, çalışmayan müşterileri tahmin etmede iyi bir performans gösterdiğini görebiliriz (1466 doğruya karşı 82 yanlış), ancak çalışan müşterileri tahmin etmede (232 doğruya karşı 328 yanlış) iyi performans göstermez.
# 
# Karar ağacı modelinin genel doğruluğuna ne dersiniz?


tr_prob2 <- predict(tr_fit, dtrain)
tr_pred2 <- ifelse(tr_prob2[,2] > 0.5,"Yes","No")
tr_tab1 <- table(Predicted = tr_pred2, Actual = dtrain$Churn)
tr_tab2 <- table(Predicted = tr_pred1, Actual = dtest$Churn)
tr_acc <- sum(diag(tr_tab2))/sum(tr_tab2)
tr_acc
# 
# 
# Karar ağacı modeli oldukça doğrudur ve %80,55 oranında test alt kümesindeki müşterilerin kayıp durumunu doğru bir şekilde tahmin eder.
# 
# Rastgele orman analizi
# Rastgele orman analizi, müşteri kaybı analizinde sıklıkla kullanılan başka bir makine öğrenimi sınıflandırma yöntemidir. Yöntem, birden fazla karar ağacı oluşturarak ve bu karar ağaçlarının özet istatistiklerine dayalı modeller oluşturarak çalışır.
# 
# Algoritmanın her bölümünde aday olarak rastgele örneklenen değişkenlerin sayısını belirleyerek başlayacağız. RandomForest paketinde buna 'mtry' parametresi veya argümanı denir.
# 



#Set control parameters for random forest model selection
ctrl <- trainControl(method = "cv", number=5, 
                     classProbs = TRUE, summaryFunction = twoClassSummary)

#Exploratory random forest model selection
rf_fit1 <- train(Churn ~., data = dtrain,
                 method = "rf",
                 ntree = 75,
                 tuneLength = 5,
                 metric = "ROC",
                 trControl = ctrl)
rf_fit


# Model, "mtry" için en uygun değerin 2 olduğunu buldu. Bu modelden, kayıp tahmin değişkenlerinin göreli önemini araştırabiliriz.


#Optimal modeli çalıştır
rf_fit2 <- randomForest(Churn ~., veri = dtrain,
                        nağaç = 75, mtry = 2,
                        önem = DOĞRU, yakınlık = DOĞRU)

#Rastgele ağaçtan değişken önemini göster
varImpPlot(rf_fit2, sort=T, n.var = 10,
           main = 'En önemli 10 değişken')

# 
# Karar ağacına benzer şekilde, bu rastgele orman modeli, sözleşme durumunu ve kullanım süresini, kayıp için önemli tahmin ediciler olarak belirlemiştir. İnternet hizmet durumu bu modelde o kadar önemli görünmüyor ve toplam ücretler değişkeni artık çok fazla vurgulanıyor.
# 
# Bu rastgele orman modelinin performansını inceleyelim. Karışıklık matrisi ile başlayacağız


rf_pred1 <- predict(rf_fit2, dtest)
table(Predicted = rf_pred1, Actual = dtest$Churn)


# 
# 
# Performans, karar ağacı modeline biraz benzer. Yanlış negatif oranı düşüktür (1445 doğruya karşı 103 yanlış), ancak yanlış pozitif oranı oldukça yüksektir (272 doğruya karşı 288 yanlış). Peki ya genel doğruluk?

# 
# 
# Rastgele orman modeli, karar ağacı modelinden biraz daha doğrudur ve test alt kümesindeki bir müşterinin kayıp durumunu %81,45 doğrulukla doğru bir şekilde tahmin edebilmektedir.
# 
# Lojistik regresyon analizi
# Son istatistiksel yöntemimiz, yukarıdaki iki makine öğrenimi tabanlı yönteme kıyasla daha klasik bir yöntem olan lojistik regresyon olacaktır. Lojistik regresyon, bir binom bağlantı işlevi kullanarak bir ikili sonuç üzerinde tahmin edici değişkenlerin geriletilmesini içerir. R, glm'deki temel genel doğrusal modelleme fonksiyonunu kullanarak modeli yerleştirelim.


lr_prob1 <- predict(lr_fit, dtest, type="response")
lr_pred1 <- ifelse(lr_prob1 > 0.5,"Yes","No")
table(Predicted = lr_pred1, Actual = dtest$Churn)

# 
# Makine öğrenimi algoritmalarına benzer şekilde, yanlış negatif oranı düşüktür (1395 doğruya karşı 153 yanlış), ancak o kadar düşük değildir. Buna karşılık, yanlış pozitif oranı (332 doğruya karşı 228 yanlış) aslında %50'nin üzerindedir, bu nedenle makine öğrenimi algoritmalarından daha iyi performans gösterir.
# 
# Genel tahmin doğruluğu, önceki modellere benzer şekilde elde edilebilir.

lr_prob2 <- predict(lr_fit, dtrain, type="response")
lr_pred2 <- ifelse(lr_prob2 > 0.5,"Yes","No")
lr_tab1 <- table(Predicted = lr_pred2, Actual = dtrain$Churn)
lr_tab2 <- table(Predicted = lr_pred1, Actual = dtest$Churn)
lr_acc <- sum(diag(lr_tab2))/sum(lr_tab2)
# lr_acc
# 
# Lojistik regresyon modelinin %81,93 doğruluk oranı, karar ağacı ve rastgele orman modellerinden biraz daha iyi performans gösterir.
# 
# Artık birkaç modele uyduğumuza ve müşteri kaybı için bazı önemli tahmin değişkenleri belirlediğimize göre, şimdi bulgularımıza dayalı olarak başka grafikleri inceleyelim.



