# Polymarket strateji araştırması — sonuç raporu (17 Eki 2026)

Veri: HF `SII-WANGZJ/Polymarket_data` zincir üstü CLOB dolumları (Eki 2025 – 4 Eki 2026), Binance spot (1 dk + 1 sn), Iowa Mesonet METAR.
Kural: her test "point-in-time" (karar anında bilinen bilgi), gerçek dolum fiyatıyla giriş, güncel taker ücreti (oran × p × (1−p)),
ilk yarıda seçim / ikinci yarıda (H2, Haziran–Ekim 2026) test, piyasa kümeli bootstrap.

## Özet tablo

| Fikir | Sonuç | Not |
|---|---|---|
| Hava favorisi (v1, 50→248 $) | Ölü | Gerçek ask ≈ fiyat + 0,5–1 sent; ücret sonrası ~0 |
| Hava maker | Ölü | Ters seçim, walk-forward −%1,2 |
| Hava tahmin modeli / METAR gözlem gecikmesi | Ölü | Piyasa tahmin ve gözlemlerden daha iyi bilgili |
| Hava "akşam kilidi" (yerel 18:00, fiyat ≥0,98) | İnce pozitif | +%0,2–0,4/işlem, kapasite medyanı ~40 $, %99,89 kazanır |
| Tüm kategoriler × vade × fiyat bandı (kalibrasyon, "bond") | Gürültü | H1/H2 hücre korelasyonu −0,15 |
| Büyük fiyat hareketine tepki | Gürültü | t < 1,1 |
| Akıllı cüzdan kopyalama | Ölü | Beceri var ama kopyalanabilir değil: tüm seçimlerde −%4..−9 |
| **Kripto up/down vs Binance spot gecikmesi** | **Tek sağlam pozitif** | H2 +%1,5/işlem, küçük kapasite, hız yarışı |

| Kripto fiyat basamakları ("Bitcoin above K", aralıklar) vs Binance modeli | Taker için ölü | Model piyasadan iyi ama taker dolumu seyrek ve negatif |
| Kim kazanıyor? | Spor | Kalıcı kazananlar spor yönlü bahisçiler (+%10–30); test için dış oran verisi gerekli (football-data engelli) |

## En iyi strateji: kripto up/down "spot gecikmesi"

Piyasalar: `btc|eth|sol|xrp|doge|bnb-updown-5m|15m-<başlangıç_unix>` ("Up or Down", çözüm Chainlink; Binance ile %95,5 uyumlu).

1. Her 10 saniyede (veya her spot güncellemesinde) her açık pencere için:
   - `x = ln(S_now / S_start)` (Binance son fiyat / pencere açılış fiyatı)
   - `sigma` = son 60 dakikanın 1 dk log-getiri standart sapması
   - `z = x / (sigma * sqrt(kalan_dakika))`
   - `P(up) = 1 / (1 + exp(-(0.0125 + 2.0696*z - 0.1696*z*|z|)))`  (z [−8, 8] aralığına kırpılır)
2. Up için `edge = P(up) − ask_up`, Down için `edge = (1 − P(up)) − ask_down`.
3. `edge > 0.02` VE `ask ≥ 0.95` ise o tarafı al (piyasa başına bir kez), çözüme kadar tut (en fazla 15 dk).
4. İşlem büyüklüğü: en iyi ask'taki miktar ile 5–20 $ arasından küçük olanı. Büyük dolumlar ters seçimli (>100 $ olanlar −%3,7).

Kanıt (%5 örnek, model H1'de kuruldu, H2 örnek dışı):
- H2: 1.025 işlem, fiyat ort. 0,955, kazanma %97,3, ROI +%1,5 (bootstrap %95: +%0,4 … +%2,5). H1: +%1,7.
- 13 ayın 11'i pozitif; 6 coinin hepsi iki yarıda da pozitif.
- Dolar ağırlıklı, işlem başı ≤5 $: H2 ROI +%1,2; ≤20 $: +%0,7; ≤50 $: +%0,1; ≤100 $: negatif.
- Tüm evrende (örnek × 20) H2'de günde ~160 fırsat. Teorik kâr ~5–8 $/gün (her yarışı kazanmak varsayımıyla). Gerçekçi beklenti birkaç $/gün.

Riskler:
- Dolumların çoğu karar anından sonraki 2 sn içinde: daha hızlı botlarla yarış. Yavaş kalırsan yalnızca geride kalan (ters seçimli olabilecek) emirleri alırsın.
- Kenar H1'den H2'ye küçüldü (rekabet artıyor); tek kötü ay (Mayıs 2026) var.
- Chainlink/Binance farkı ve sınır durumları (%4,5 uyumsuzluk, büyük z'de nadir).
- Canlı çalıştırma Polymarket CLOB erişimi ister; bu bağlantıdan API'ye erişilemiyor (çözülmesi kullanıcıya ait).

## Önerilen sonraki adım (gerçek para öncesi)
1. Kağıt üstünde ileri test (2–4 hafta): canlı defteri (`book` uç noktası) ve Binance websocket'ini kaydet, kuralı gerçek ask/derinlikle simüle et, gecikmeyi ölç.
2. Kâğıt test pozitifse: 5 $/işlem, günlük zarar limiti 10 $, 100 $ sermaye ile küçük canlı test.
3. Kenar 2–3 hafta boyunca %0,5'in altına düşerse dur.

Kodlar: `src/a13_binance_fetch.py`, `src/a13c_crypto_1s.py`, `src/a13d_breakdown.py`, `src/a13e_capacity.py`. Ayrıntılar `CLAUDE.md`.
