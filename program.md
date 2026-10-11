# autoresearch: NBA spread / total

LLM, bir River modeliyle NBA bahis çizgisine karşı avantaj arar.

## Kurulum

1. Bugünün tarihinden bir etiket seç (ör. `oct11`) ve main'den `git checkout -b autoresearch/<etiket>` çalıştır. Branch yeni olmalı.
2. Şunları oku: `prepare.py` (veri ve değerlendirme; değiştirme), `train.py` (değiştirebileceğin tek dosya), `nba_2008-2026.csv` dosyasının başlık satırı ve `divisions.json`.
3. `results.tsv` dosyasında yalnızca başlık satırı var. İlk koşu, dokunulmamış baseline'dır.

## Görev

CSV'deki her satır bir maç ve o maçın kapanış çizgisidir. İki hedef var; değerleri 1 / 0 / 2 (2 = push):

- `id_spread`: favori (`whos_favored`) `spread`'i kapattı.
- `id_total`: toplam skor `total`'ın üstünde bitti.

İkisi de yazı-tura sayılır. -110 oranında başabaş yaklaşık %52.4'tür.

`prepare.py`, her hedef için River `progressive_val_score`'u ayrı ayrı çalıştırır. Etiket 1 gün gecikmeli açılır ve iki hedef ortak bir `Accuracy` + `LogLoss` metriğine yazılır. Push skorlanmaz.

`train.py` içindeki `make_model(target)` bir River sınıflandırıcı döndürür. `x` maç öncesi satırdır, `y` hedef tuttuysa `True` olur. Skorlar modele hiç verilmez. Maçtan maça değişen durum (dinlenme günleri, çizgi seviyeleri) bir transformer'da tutulur ve yalnızca önceki günlerin maçlarıyla güncellenir (bkz. Veri sızıntısı).

## Kurallar

- Yalnızca `train.py`'yi değiştir ve yalnızca River kullan. Paket kurma, `train.py` içinden CSV okuma.
- Hedef en yüksek `val_acc`. Eşitlikte düşük `val_logloss` kazanır.
- Standart hata yaklaşık 0.002'dir; +0.001 gürültüdür.
- Basit kod kazanır. Bir şeyi silip skoru koruyabiliyorsan bu bir kazançtır.
- Mikro ayar yapma: sabitlerle oynama, seed avlama. Bunun yerine başka bir model ya da özellik dene.
- Kurallar dahilinde yenilikci seyler deneyebilirsin.

## Veri sızıntısı

Sızıntıyla gelen skor geçersizdir; böyle bir commit tutulmaz. Her değişiklikte şunları kontrol et:

- Bir maçın tahmini yalnızca maçtan önce bilinenlere dayanır: `x`'teki maç öncesi kolonlar ve önceki günlerin maçları. Skorlar, devre arası çizgileri ve aynı günün diğer maçları (sonuçları, çizgileri, sırası, sayısı) kullanılmaz.
- River `Pipeline.learn_one`, transformer'ın `learn_one(x)` metodunu `transform_one(x)`'ten önce çağırır. Durumu `learn_one`'da hemen güncellersen öğrenilen özellik maçın kendisini görür (ör. dinlenme hep 0 olur). `train.py`'deki bekletme düzenini kullan: `learn_one` maçı `pending`'e ekler, `transform_one` önceki günlerin maçlarını `update_state` ile işler. Bir maç tahminde ve öğrenmede aynı özelliklerle dönüştürülmelidir.
- Sabitler o tarihte bilinebilecek bilgiler olmalıdır (ör. sezon takvimi). Veriye bakarak seçilmiş eşik, takım, sezon ya da tarih listesi yazma.
- `predict_proba_one` her çağrıda yeni bir sözlük döndürür. Döndürülen tahmini saklayıp sonradan değiştirme.
- Beklenmedik büyük bir kazanç (ör. +0.005'ten fazla) önce sızıntı şüphesiyle incelenir: özelliklerin tahmin anındaki ve öğrenme anındaki değerlerini karşılaştır.

## Sonsuz döngü

1. `train.py`'yi değiştir ve `git commit` yap.
2. `.venv/Scripts/python train.py > run.log 2>&1` çalıştır (600 sn sınırı).
3. `grep "^val_acc:\|^val_logloss:" run.log` çalıştır. Çıktı boşsa koşu çökmüştür: `tail -n 50 run.log` ile oku. Hata basit bir yazım hatasıysa düzelt, değilse fikri bırak.
4. `results.tsv`'ye bir satır ekle (sekmeyle ayrılmış, commit edilmez):
   `commit  val_acc  val_logloss  keep|discard|crash  açıklama`
5. `val_acc` arttıysa commit'i tut. Artmadıysa `git reset --hard` ile son tutulan commit'e dön.
6. Geçici dosyaları sil.

Hiç durma ve devam edip etmeyeceğini sorma. Sadece insan durdurunca dur.
