# Polymarket hava durumu stratejisi araştırması

Kullanıcı Türkçe konuşuyor; yanıtları Türkçe ver.

## Amaç
Polymarket günlük sıcaklık piyasalarında ("Highest temperature in <city> on <date>?") stratejiyi derinleştirmek ve yeni stratejiler keşfetmek. Önce backtest, sonra ileriye dönük test; gerçek para en son.

## Baz strateji (v1)
- Karar anı: hedef günün yerel gece yarısından 24 saat önce.
- Her şehir/gün için en yüksek YES fiyatlı sıcaklık aralığı (favori) seçilir (karar öncesi son 120 sn fiyatı).
- Alış fiyatı = fiyat + 0,01; 0,60 <= alış < 0,90 ise al. Pay başı 1 $ öder.
- 50 $ başlangıç, işlem başı en fazla 5 $, min 5 pay, açık işlemin parası çözülene kadar kilitli.

## v1 sonucu (15 Eki 2025 – 16 Eki 2026, `src/backtest_csv.py`)
- 11.437 etkinlik, 505 işlem, 383 kazanan (%75,8), 50 → 248,60 $, getiri %7,9 (bootstrap %95: %2,1–%13,2).
- Aylık net: Kas -5,0; Ara +27,1; Oca -1,5; Şub +10,3; Mar +48,6; Nis +17,0; May +59,9; Haz +13,8; Tem -10,2; Ağu +25,8; Eyl +24,7; Eki(16'ya kadar) -11,6.
- Kırılganlık: +3 sent kaymada getiri %1,7; kârın %56'sı alış 0,60–0,65 bandında; kârın %39'u 3 şehirde (Tel Aviv, Lucknow, İstanbul).
- Tüm favoriler (banttan bağımsız): fiyatı 0,6–0,7 olanların %70,7'si, 0,7–0,8 olanların %80,9'u kazandı (piyasa favoriyi hafif düşük fiyatlıyor olabilir).
- Ayrıntı: `results/backtest_v1_results.json`.

## Analiz bulguları (17 Eki 2026, `src/analysis_lib.py` + `src/a1..a3_*.py`, sonuçlar `results/a*.json`)
- `analysis_lib.py` v1'i bağımsız yeniden uygular ve aynı sonucu verir (505 işlem, 248,60 $). Yeni analizler bunu kullanır; `PYTHONPATH=src python src/a1_sensitivity.py`.
- A1 kayma/ücret: başa baş kayma ~4 sent/pay (sabit 5 $ stake, ücret 0). 2 sent -> ROI %4,5; 3 sent -> %1,7. Tarih/hafta/ay kümeli bootstrap CI'yi genişletmiyor (~%2,6–%12,8).
- A2 walk-forward (bant geçmiş aylarda seçilir, sonraki ayda test, Oca–Eki 2026): 1 sent kaymada OOS ROI %4,8–7,4; 2 sentte %2,1–3,8. İlk 6 ay seç/son 6 ay test: 0,60–0,75 bandı, ROI %6,8 (1 sent) / %3,5 (2 sent).
- A3 kalibrasyon: favoriler 0,30–0,95 boyunca düşük fiyatlı (0,40–0,50 bandı n=2588: fiyat %44, kazanma %47,3). Etkinlik başına fiyat toplamı ~1,05–1,07. Uç aralıklara NO almak maliyet sonrası ~0. Şehir farkları ~±5 puan standart hata içinde -> büyük ölçüde gürültü.
- Mart 2026'dan sonra piyasaların ~%78'i °C, favori medyan fiyatı 0,35–0,42; v1 bandı (0,60–0,90) etkinliklerin yalnızca %2–6'sına uyuyor.
- Bu bilgisayardan Polymarket API'leri (gamma/clob/data-api) erişilemez (Türkiye engeli); Open-Meteo erişilebilir. Polymarket verisi bulut agent'larıyla çekilir.

## Araştırma organizasyonu (tüm Polymarket)
Tek hedef: maliyetler sonrası örnek dışında para kazandıran strateji. Bulut agent'ları ayrı dallarda çalışır, `results/<iş>/summary.md` üretir:
`analysis-fill-check` (hava piyasası gerçek dolgu fiyatı + çözüm istasyonları), `research/calibration` (tüm kategorilerde kalibrasyon, "bond" işlemi), `research/crypto` (kısa vadeli kripto piyasaları vs spot model), `research/sports` (spor vs Pinnacle kapanış oranları), `research/structural` (negRisk/merdiven arbitrajı, canlı defter taraması), `research/weather-forecast` (Open-Meteo tahmin modeli).
Ortak protokol: gerçekçi giriş (taker işlemleri/defter), güncel ücretler, walk-forward, kümeli bootstrap, kapasite; olumsuz sonuç da raporlanır. Sadece okuma: emir, cüzdan, API anahtarı yok.

## KRİTİK sınırlamalar
- CLOB `prices-history` serisi son işlem/orta fiyattır, GERÇEK ask değil. Dolgu fiyatı doğrulanmadı. Gerçek ask/derinlik toplanmadan sonuç iyimser kabul edilmeli.
- Bant (0,60–0,90) ve 24 saat kuralı eski bir Codex sohbetinden miras; aynı veride seçildiği için örnek içi sonuç şişkin olabilir. Walk-forward / out-of-sample henüz yok.
- Ücret modeli varsayımsal (pay x oran x p x (1-p)).
- Eski Codex sonuçları (50→77,58; 50→30,84; walk-forward 50→48,28) farklı dönem/kapsamdı, bununla uzlaştırılmadı.

## Veri
- `data/decision_prices.csv.gz`: her piyasa için karar anındaki son fiyat (price_last_le_dec), yaşı (sn), sonuç (outcome_yes), kapanış zamanı. Karar anından SONRAKİ fiyatlar yok.
- Ham veri (790 MB, git'te yok) yeniden üretmek için (~1 saat, `data/raw/` içinde çalıştır):
  `python3 ../../src/list_events.py` -> events.json; `python3 ../../src/list2.py` (weather etiketi, Eki 2025–Şub 2026 eksiklerini ekler; events.json + events2.json -> events_all.json birleştir); `python3 ../../src/fetch_px.py` (events_all.json okur) -> px.jsonl; `src/export_compact.py` CSV'yi üretir.
- Gamma API: `https://gamma-api.polymarket.com/events?tag_slug=highest-temperature&closed=true&end_date_min=..&end_date_max=..` (günlük döngüyle sayfala). CLOB: `https://clob.polymarket.com/prices-history?market=<yes_token>&startTs&endTs&fidelity=1`.
- Ekim 2025–Ocak 2026 etkinlikleri `weather` etiketi altında, `highest-temperature` etiketinde yok.
- Şehir saat dilimleri `src/common.py` içinde.

## Araştırma gündemi
1. Walk-forward: kuralları (bant, karar saati) geçmiş pencerede seç, sonraki pencerede test et.
2. Karar saatini değiştir (48/12/6/2 saat önce): bunun için karar öncesi geniş fiyat geçmişi çekilmeli (`fetch_px.py` penceresini genişlet).
3. Favori dışı stratejiler: kalibrasyon farkı (model olasılığı vs piyasa fiyatı), komşu aralık spread'i, uç aralıklara NO almak, şehir/mevsim bazlı filtre.
4. Hava tahmini ile karşılaştırma (Open-Meteo vb. ücretsiz tahmin API'si) ve piyasa fiyatından sapma.
5. Gerçek ask/derinlik kaydı: CLOB `book` uç noktasını periyodik poll et, dolgu simülasyonunu gerçek veriyle yap.
6. Risk: Kelly/sabit oran, şehir bazlı limit, eşzamanlı açık pozisyon limiti.
