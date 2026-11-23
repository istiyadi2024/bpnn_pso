# Eksperimen v2 — BPNN vs PSO untuk Prediksi Harga Emas (protokol jurnal)

Folder ini berisi eksperimen ulang dari skripsi dengan protokol yang layak untuk
jurnal (target: IJEECS). Semua angka di bawah berasal dari `results/` dan
`results_return/`.

## Cara menjalankan

```
pip install numpy pandas scipy statsmodels matplotlib
python run_experiment.py                 # skenario A: target = level harga
set TARGET=return && python run_experiment.py   # skenario B (Windows CMD); bash: TARGET=return python run_experiment.py
python make_figures.py                   # grafik skenario A
set TARGET=return && python make_figures.py     # grafik skenario B
```
Durasi sekitar 2 menit per skenario. `dataset.csv` dibaca dari folder induk.

## Apa yang diubah dibanding skripsi

| Aspek | Skripsi | Eksperimen v2 |
|---|---|---|
| Fitur | bulan, tahun, minyak, kurs (bulan yang sama) | lag: emas t-1..t-3, minyak t-1, kurs t-1 (benar-benar *forecasting*) |
| Pembagian data | acak 80/20 | kronologis: latih Jun 2013–Feb 2020 (81), validasi Mar 2020–Agu 2021 (18), uji Sep 2021–Feb 2023 (18) |
| Normalisasi | di-*fit* ke seluruh data | di-*fit* hanya ke data latih |
| Pemilihan parameter | berdasarkan data uji | berdasarkan data validasi; data uji dipakai sekali |
| Jumlah run | 1 | 30 run independen (seed 1–30) |
| Metrik | RMSE ternormalisasi | RMSE, MAE, MAPE, R² dalam USD/oz |
| Pembanding | BPNN | Naive (random walk), ARIMA, MLR, BPNN, PSO-NN, GA-NN, PSO-BP |
| Uji statistik | — | Friedman, Wilcoxon (koreksi Holm), Diebold-Mariano (HLN) |
| PSO | c1 = c2 = learning rate; **bug pbest/gbest** | c1 = c2 = 1,49445, kecepatan dibatasi, pbest/gbest disalin dengan benar |

### Bug pada `hybridnn` (penting untuk diketahui)
Di `Particle.evaluate()` tertulis `self.best_position = self.position`, dan di
`NNPSO.optimize()` tertulis `self.global_best_position = particle.position`.
Keduanya menyimpan *referensi* ke array yang sama, bukan salinan. Karena
`update_position()` mengubah array secara in-place, pbest dan gbest ikut
bergeser setiap iterasi, sehingga bobot "terbaik" hilang. Tes kecil menunjukkan
RMSE gbest yang tercatat 0,264, padahal RMSE bobot yang benar-benar dipakai
untuk prediksi 0,386.
Perbaikannya: `self.best_position = [p.copy() for p in self.position]`, dan hal
yang sama untuk `global_best_position`.

Dengan kode asli dan konfigurasi terbaik skripsi (50 partikel, w = 0,6, lr = 0,7),
30 run memberi RMSE NNPSO 301 ± 130 USD (rentang 96–586). NNPSO hanya
mengalahkan BPNN pada **2 dari 30 run**. Artinya, hasil "eror turun 20,64%" di
skripsi adalah satu run yang kebetulan baik (`results/fig6_protocol_check.png`).

## Hasil utama

### Skenario B — target log-return (DISARANKAN untuk naskah)
Nilai harga direkonstruksi: harga_t = harga_{t-1} · exp(return prediksi).
Konfigurasi terpilih dari validasi: hidden = 5, lr BP = 0,1, PSO 100 partikel dengan inertia menurun 0,9 → 0,4.

| Model | RMSE (USD) mean ± sd | MAPE (%) | R² |
|---|---|---|---|
| Naive / random walk | 65,74 | 3,13 | 0,352 |
| ARIMA(0,1,0) (dipilih AIC) | 65,74 | 3,13 | 0,352 |
| MLR | 67,38 | 3,24 | 0,319 |
| BPNN | 68,01 ± 1,88 | 3,38 | 0,306 |
| **PSO-NN** | **65,71 ± 2,74** | **3,10** | **0,352** |
| GA-NN | 66,03 ± 3,25 | 3,12 | 0,345 |
| PSO-BP | 66,36 ± 3,28 | 3,21 | 0,338 |

* Friedman: χ² = 17,48, p = 0,0006, artinya ada perbedaan antar metode pelatihan.
* PSO-NN lebih baik dari BPNN pada 24/30 run (Wilcoxon, p-Holm = 0,004). Penurunan RMSE rata-rata 3,4%.
* GA-NN juga lebih baik dari BPNN (p-Holm = 0,004). PSO-NN dan GA-NN tidak berbeda signifikan.
* **Tidak ada model yang signifikan lebih baik dari random walk** (semua uji DM p > 0,5). Ini konsisten dengan hipotesis pasar efisien untuk horizon bulanan dan harus dilaporkan secara jujur.

### Skenario A — target level harga (untuk diskusi atau lampiran)
Semua NN lebih buruk dari random walk (BPNN 96,9 ± 24,9; PSO-NN 130,6 ± 41,1 USD).
Penyebabnya: data uji (1.630–1.940 USD) berada di luar rentang data latih
(maksimum sekitar 1.600 USD), dan NN tidak bisa berekstrapolasi. Ini alasan
metodologis untuk memakai log-return, dan bisa ditulis satu paragraf di Discussion.

## Implikasi untuk naskah IJEECS
1. Klaim yang bisa dipertahankan: *"PSO-based training yields a statistically
   significant improvement over gradient-based BP training (24/30 runs,
   Wilcoxon p<0.01), but neither model significantly outperforms the random-walk
   benchmark (DM test)"*. Reviewer menghargai kejujuran ini lebih dari klaim
   "eror turun 20%" yang tidak bisa direplikasi.
2. Klaim penghematan waktu 37% sebaiknya dihapus. Dengan protokol adil, PSO-NN
   (0,19 s) tidak lebih cepat dari BPNN (0,11 s), dan perbedaan sepersekian detik
   tidak bermakna.
3. Agar hasilnya lebih kuat, tambahkan data hingga 2025/2026, ganti USD/IDR
   dengan DXY, dan pertimbangkan data mingguan (lebih banyak sampel).
   Jalankan ulang skrip yang sama.

## Isi folder
* `nnopt.py`: model (BPNN, PSO-NN, GA-NN, PSO-BP), hanya butuh numpy
* `run_experiment.py`: seluruh protokol eksperimen
* `make_figures.py`: grafik 300 dpi untuk naskah
* `results/`, `results_return/`: CSV hasil, uji statistik, dan grafik
  * `summary_nn.csv`, `summary_baselines.csv`: tabel utama
  * `stats_wilcoxon.csv`, `stats_dm.csv`: uji statistik
  * `sensitivity_pso.csv`, `fig5_pso_sensitivity.png`: analisis sensitivitas swarm × inertia
  * `runs_final.csv`: 30 run per model
  * `protocol_check_*`: efek protokol skripsi dan bug asli (hanya di `results/`)
