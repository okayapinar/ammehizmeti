# autoresearch: NBA spread / total

Suna bi el atin olmuslerinizin rahmetine. 

Bir LLM ajanı, NBA maçlarında bahis çizgisine karşı sonucu (spread kapandı mı, total üstü mü) tahmin eden modeli kendi kendine geliştirir. Hedef: 2024+ sezonlarında en yüksek `val_acc` (başabaş ~%52.4).

Fikir ve yapı Andrej Karpathy'nin [autoresearch](https://github.com/karpathy/autoresearch) projesinden uyarlandı. Orijinalde ajan, küçük bir GPT eğitimini (nanochat) tek GPU'da 5 dakikalık bütçeyle iyileştirmeye çalışır. Burada aynı döngü (`program.md` talimatları, tek değiştirilebilir `train.py`, sabit `prepare.py`, `results.tsv` kaydı) NBA bahis tahminine uygulandı.

- `train.py`: model, ajanın değiştirdiği tek dosya
- `prepare.py`: veri ve sabit değerlendirme, değiştirilmez
- `program.md`: ajanın talimatları
- `agent.py`: ajanı sonsuz döngüde çalıştırır
- `results.tsv`: deney kayıtları

Revision: No ML magic matters, the model cannot lower the noise floor, as that's the optimal bound of Shannon encoding of your data. (Cok yormamak lazim, olmuyorsa zorlamamak gerekir.)
