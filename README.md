# Final Project Kelompok 1

## Git Hub URL
<https://github.com/amelianisrinah/kelompok1.git>

## Background
Dataset Introduction & Problem Statement :

Dataset SBAnational.csv adalah dataset yang berasal dari U.S. Small Business Administration (SBA), yang berperan membantu pihak pengusaha kecil di pasar kredit AS melalui program pemberian jaminan yang dirancang agar pihak Bank dapat memberikan pinjaman kepada pengusaha kecil. SBA bertindak seperti penyedia asuransi untuk mengurangi risiko pihak Bank dengan menutupi beberapa kerugian melalui sebagian jaminan yang diberikan jika para pengusaha kecil gagal bayar, sehingga pihak Bank tidak perlu khawatir.
Namun karena jaminan yang diberikan SBA hanya sebagian dari seluruh saldo pinjaman, Bank akan mengalami kerugian jika pihak pengusaha kecil gagal dalam melunasi pinjaman yang dijamin SBA. Oleh karena itu, Bank masih dihadapkan pada pilihan yang sulit apakah mereka harus memberikan pinjaman tersebut atau tidak karena harus mempertimbangkan risiko gagal bayar yang masih tinggi walaupun sebagian sudah dijamin oleh pihak SBA.

Role :

Disini kami berperan sebagai Tim Data Scientist dari pihak Bank

Goals :

Adapun goals yang ingin kami capai adalah meningkatkan efektivitas dalam melakukan review terhadap calon peminjam tanpa menambah cost untuk menambah karyawan, serta meminimalisir risiko adanya pengusaha kecil yang tidak mampu melunasi pinjaman.

Business Matrics :

Berikut adalah Business Matrics yang kami gunakan sebagai pengukur keberhasilan dari goals yang ingin kami capai :
1. Daily resolved application (banyaknya pengajuan yang berhasil di-review per hari), tanpa menambah cost untuk menambah karyawan dalam melakukan review dari pengajuan pinjaman yang masuk.
2. Failed pay ratio (rasio gagal bayar dari total pinjaman yang diterima), yakni meminimalisir risiko adanya pengusaha kecil yang tidak mampu melunasi pinjaman
Objective :

Berdasarkan latar belakang yang ada kami bermaksud melakukan analisa pada data historis yang tersedia, untuk membuat Machine Laerning Model yang dapat memprediksi secara otomatis penilaian risiko pinjaman, apakah pinjaman yang diajukan oleh pengusaha kecil layak untuk disetujui atau ditolak. Sehingga dengan adanya model ini, daily resolved application akan meningkat karena banyak pengajuan yang di-review secara otomatis oleh model, serta meminimalisir risiko gagal bayar dari pengusaha kecil.

## Stage 1 (EDA)

#### Check Null Values
![image](https://user-images.githubusercontent.com/114790120/193457624-78c4a3c8-f78f-4933-a1c1-df48db18678e.png)

**Null Value :**
- Terdapat 11 kolom yang memiliki Null Values:
  - *Name, City, Satate, Bank, BankState, NewExist, RevLineCr, LowDoc, ChgOffDate, DisbursementDate, MIS_Status*
- Kolom yang mempunyai nilai kosong paling banyak adalah kolom ChgOffDate dengan jumlah 736,465 yakni sekitar 82% dari total seluruh baris.
- Persentase null values pada dataset sangat tinggi yaitu 83.6% hal tersebut karena feature 'ChgOffDate' memiliki jumlah null values sangat tinggi sehingga perlu diperiksa lebih lanjut.
- Apabila 'ChgOffDate' didrop maka null valuesnya hanya 1.64% sehingga 1 variabel ChgOffDate ini bisa didrop saja.
- Null values pada kolom 'MIS_Status' sebaiknya didrop karena kolom tersebut merupakan target sehingga tidak boleh ada nilai yang kosong

#### Change Data Type
- **Object to DateTime**
['ApprovalDate','DisbursementDate','ChgOffDate']
- **Object to Numeric**
['DisbursementGross', 'BalanceGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv']
- **Float to Int**
['NewExist']
- **Numeric to Object**
['NAICS']

#### Statistical Summary

**Descriptive Statistic Time**
![image](https://user-images.githubusercontent.com/114790120/193458306-3f3962e0-8541-4d74-974f-724f7f90498c.png)

**Observation Result:**
1. 75% data pada feature **ChgOffDate** berisi null value, feature ini menunjukkan tanggal ketika peminjaman dinyatan gagal maka ketika peminjam berhasil dibayar kolom tersebut nilainya null. Feature ChgOffDate lebih baik didrop nantinya, karena kita bisa berfokus dengan kolom MIS_Status status saja.
2. **ApprovalDate** dan **ApprovalFY** merujuk pada waktu pinjaman disetujui, kedua features ini tidak dapat menjadi preditor target sehingga sepertinya akan didrop. Selain itu pada feature ApprovalDate terdapat anomali waktu terakhir yang menunjukkan tahun 2071.
3. **DisbursementDate** menunjukkan tanggal pinjaman dicairkan. Terdapat anomali pada data waktu awal dan akhir. Menurut dokumentasi data yang ada pada dataset hanya diambil dari tahun 1987-2014, kemungkinan ada salah input pada datanya sehingga perlu diperiksa sebaran datanya.

**Descriptive Statistic Numerical**
![image](https://user-images.githubusercontent.com/114790120/193458160-d646deb8-fba2-46f8-9ff7-66607d17cabe.png)

**Observation Result :**
- Dilihat dari jarak nilai median dan rata-rata pada keseluruhan variabel dapat disimpulkan bahwa datanya skewed.
- Dari nilai Max/Min keseluruhan variabel, dapat disimpulkan hampir seluruh variabel memiliki outliers.

- **Term** (jangka waktu pinjaman dalam bulan)
  - Kolom Term adalah jangka waktu pinjaman dalam bulan, jika dilihat dari nilai minimunnya adalah 0 sedangkan untuk jangka waktu pinjaman seharusnya tidak ada yang dibawah 1 bulan.
  - Pada dokumen dataset disebutkan terdapat pinjaman yang didukung oleh Real Estate (dijadikan agunan) yg dapat dijadikan indikator resiko pemberian pinjaman. Alasan untuk indikator ini adalah bahwa nilai Real Estate cukup besar untuk menutupi jumlah pinjaman, sehingga mengurangi kemungkinan gagal bayar. Pada dokumen dataset pinjaman yang didukung oleh real estate akan memiliki jangka waktu 20 tahun atau lebih (???240 bulan). Untuk itu feature term dapat digunakan untuk membuat variabel dummy yg menunjukkan apakah pinjaman didukung oleh real estate atau tidak. 
  - 75% data memiliki term (jangka waktu pinjaman) diatas 5 tahun, hal ini sesuai dengan dokumentasi bahwa pinjaman sering dalam waktu 5 tahun atau lebih. Pada bagian Descriptive Statistics - Time yakni fitur DisbursmentDate, telah diketahui bahwa data yang tersedia pada dataset seharusnya dari tahun 1987 s.d 2014, sehingga untuk pinjaman yang dicairkan diatas 2009 jangka waktu minimum pinjaman ada diatas tahun 2014. Selain itu dapat dilihat anomali pada nilai max yang ada diangka 569 bulan (sekitar 47 tahun), sedangkan data yang ada dari tahun 1987 s.d 2014 hanya tersedia 27 tahun, kemungkinan ada *value* yang *error* diatas tahun 2014 dan di. Analisa selanjutnya adalah apabila kami mengambil data 2010 s.d 2014, maka akan memberatkan di status CHGOFF, karena pada dokumentasi disebutkan bahwa dataset ini merupakan data yang sudah pasti statusnya. Dari sini dapat diasumsikan bahwa data CHGOFF sudah dapat dideteksi statusnya sedangkan data yang belum memiliki status tidak dimasukan pada dataset ini. Berlatarbelakang hal tersebut kami memutuskan untuk mengambil data dari tahun 1987 s.d 2009 saja. **Untuk analisis selanjutnya kami memutuskan untuk menggunakan data dari tahun 1987 s.d 2009.**

- **NoEmp** (Jumlah karyawan yang dimiliki) <br>
Pada variable NoEmp nilai minimunya adalah 0, sedangkan NoEmp adalah jumlah karyawan bisnis yang dimiliki, dan tidak mungkin jika sebuah perusahaan tidak memiliki karyawan sama sekali. Kemungkinan ada kesalahan dalam input datanya.

- **RetainedJob**
 - Feature ini memiliki 25% data bernilai 0.
 - Dari feature ini lebih baik dibuat feature baru yg menunjukkan apakah bisnis memiliki karyawan tetap atau tidak.

-  **BalanceGross dan ChgOffPrinGr** <br>
75% data memiliki nilai 0 sehingga dua features ini akan didrop saja.

- **SBA_Appv** <br>
SBA_Appv menunjukkan jumlah jaminan yang diberikan oleh SBA kepada bank. Sebaiknya value dari feature ini dibuat dalam bentuk persentase.

- **DisbursementGross** <br>
Terdapat anomali pada nilai minimum yaitu 0, dimana hal ini tidak mungkin.


**Descriptive Statistic Categorical**
![image](https://user-images.githubusercontent.com/114790120/193458214-d6755924-417a-47b4-9324-d1f843062665.png)

**Observation Result :**

- **LoanNr_ChkDgt, Name, Bank**, mempunyai kardinalitas yang tinggi, dan sepertinya tidak berpengaruh terhadap target sehingga akan didrop.

- **City** dan **Zip** mempunyai kardinalitas yang tinggi dan dapat digantikan oleh feature State sehingga kedua features ini akan didrop.

- **State** dan **BankState** memiliki unique value yang cukup banyak, akan lebih mudah apabila dilakukan feature engineering dengan membuat feature baru yg menunjukkan state peminjam apakah sama dengan state bank untuk dilihat pengaruhnya terhadap target. Selain itu perlu dilihat berapa default rate pada masing-masing State.

- Pada dokumentasi dataset disebutkan bahwa dua digit pertama pada kode **NAICS** menunjukkan industri dari bisnis yg dijalankan oleh peminjam. Berdasarkan hal ini akan dibuat kolom baru untuk mengklasifikasikan NAICS berdasarkan nama sektor industrinya.

- **NewExist** terdeteksi memiliki 3 *unique value*, sedangkan seharusnya hanya ada 2 *unique value* yakni, 
1 = *Existing business*,
2 = *New business*

- Feature **FranchiseCode** menunjukkan kode franchise dimana nilai 00000 atau 00001 = No franchise, jadi sebaiknya kolom ini dibuat *boolean value*.

- **RevLineCr** terdeteksi memiliki 18 unique values, sedangkan seharusnya hanya ada 2 unique value yakni, 
Revolving line of credit: 
Y = Yes, N = No
kemungkinan ada kesalahan input.

- **LowDoc** terdeteksi memiliki 8 unique values, sedangkan seharusnya hanya ada 2 unique value yakni, 
LowDoc Loan Program: Y = Yes, N = No
kemungkinan ada kesalahan input. Berdasarkan dokumentasi dataset feature ini menunjukkan jumlah dokumen yang diperlukan berdasarkan besaran pinjaman dimana ketika nilai pinjaman < 150.000 dokumen yg diperlukan sedikit (LowDoc='Y') sementara ketika pinjaman >= 150.000 dokumen yg diperlukan lebih banyak (LowDoc='N').

- **MIS_Status** <br>
 - Terdiri dari 2 unique values yaitu:
   * P I F = Paid in full, yakni pinjaman yang dibayar lunas
   * CHGOFF = Loan status charged off, yakni pinjaman yang gagal dibayarkan
 - Total value sebanyak 897033 dengan value didominasi oleh P I F sebanyak 739489. Dari sini dapat disimpulkan bahwa target imbalace sehingga perlu dilakukan over sampling atau under sampling saat pre-processing nantinya.

#### Univariate Analysis
**Observation Result :**

**- Date Time Features**
Terdapat banyak nilai error yang menyebabkan distribusi datanya positively skewed. sehingga kami memutuskan untuk mengambil data sesuai dengan analisa pada bagian descriptive term sebelumnya yakni dari tahun 1987 s.d 2009. **Untuk analisis-analisis selanjutnya kami hanya akan melakukan analisa pada data untuk data tahun tahun 1987 s.d 2014.**

**- Numerical Features**
Terlihat outlier dengan distribusi positively skewed pada seluruh features, sehingga harus dilakukan handling outliers saat data pre-processing.

**- Categorical Features**
1. Dapat dilihat bahwa NewExist, RevLineCr dan LowDoc memiliki nilai error.
2. NewExist, MIS_Status dan LowDoc imbalance. 

#### Multivariate Analysis
**Observation Result :**

**Numerical Features**
- **Term** memiliki korelasi yang positif dengan **DisbursementGross, Gross Approved**, dan **SBA_Appv**. Hal tersebut mengindikasikan jangka waktu pinjaman yang lebih lama berpotensi memiliki nilai pinjaman yang lebih besar.
- **DisbursementGross, GrAppv,** dan **SBA_Appv** memiliki korelasi yang kuat satu sama lain. Ada kemungkinan bahwa ketiganya redundant sehingga perlu dipilih salah satu saja apabila akan dijadikan feature.
- **DisbursementGross, GrAppv,** dan **SBA_Appv** memiliki korelasi dengan **ChgoffPrinGr**. Hal tersebut masuk akal karena semakin besar nilai pinjaman yang diberikan, maka apabila terjadi Chgoff atau gagal bayar nilainya Chgoff yang diwakili ChgoffPrinGr akan lebih besar
- **CreateJob** dan **RetainedJob** sama-sama memiliki korelasi 0.99 bersifat redundant sehingga hanya dipilih salah satu saja apabila akan dijadikan feature.
- **DisbursementGross** berkolerasi positif dengan **NoEmp** hal ini menunjukkan bahwa semakin banyak jumlah karyawan (indikasi perusahaan yang lebih besar) maka semakin besar jumlah pinjaman yang cair.

**Anova Test Numerical to Categorical (Target)**
- Pvalue Term: 0.0

Term mampu membedakan PIF dan CHGOFF
- Pvalue NoEmp: 1.1848563979530901e-136

NoEmp mampu membedakan PIF dan CHGOFF
- Pvalue CreateJob: 1.8043882079552313e-29

CreateJob mampu membedakan PIF dan CHGOFF
- Pvalue RetainedJob: 1.2789476431558583e-31

RetainedJob mampu membedakan PIF dan CHGOFF
- Pvalue DisbursementGross: 0.0

DisbursementGross mampu membedakan PIF dan CHGOFF
- Pvalue GrAppv: 0.0

GrAppv mampu membedakan PIF dan CHGOFF
- Pvalue SBA_Appv: 0.0

SBA_Appv mampu membedakan PIF dan CHGOFF

## Stage 2 (Data Pre Processing)

#### Data Cleansing
**Handling Missing Values**
Terdapat 1.6% null values (di luar chgoffdate) dari keseluruhan data maka hal yang terbaik dilakukan adalah dengan mendrop baris dengan null values daripada melakukan imputasi. Berikut beberapa kolom yang memiliki null values:
- 'State'
- 'BankState'
- 'LowDoc'
- 'RevLineCr'
- 'DisbursementDate'
- 'MIS_Status'

**Handling Error Values**
Beberapa kolom yang memiliki error values:
- 'NewExist'
- 'RevlineCr'
- 'Term'
- 'Noemp'

**Handling Duplicated Data**
Tidak terdapat data duplicate

#### Feature Encoding
- 'NewExist' 
- 'RevlineCr'
- 'IsDefault' dari kolom 'MIS_Status'
- 'IsRetainedJob'

#### Feature Enginering

**Feature Extraction**
- 'Real Estate', Apakah pinjaman memiliki term lebih dari 240 bulan
- 'SBA_AppvPct', Persentase SBA Approved dari Gross Approved
- 'StateSame', Apakah lokasi perusahaan dan bank pinjam sama
- 'DefRateState', Default rate per state
- 'Industry', Industri mengacu pada kode NAICS
- 'DefRateIndustry', Default rate per industry
- 'IsFranchise', Apakah perusahaan peminjam merupakan Franchise
- 'Recession', Apakah perusahaan aktif ketika masa resesi

**Feature Selection**
Terdapat beberapa fitur yang didrop karena tidak relevan, yaitu:
['LoanNr_ChkDgt', 'Name', 'City', 'State', 'Zip', 'Bank', 'BankState','NAICS', 'ApprovalDate', 'ApprovalFY','FranchiseCode', 'UrbanRural',
'ChgOffDate', 'DisbursementDate', 'DisbursementGross', 'BalanceGross', 'MIS_Status', 'ChgOffPrinGr','SBA_Appv','Industry']

**Handling Outliers**
Handling outliers dilakukan pada 'Term','DisbursementGross'

**Standardization**
Melakukan standarisasi pada fitur yaitu 'BCTerm','SBA_AppvPct','DefRateState','DefRateIndustry','BCDisbursementGross'

**Handling Imbalance**
Menggunakan Oversampling SMOTE 0,5

## Modelling
Kami memiliki class yang imbalance, meskipun telah dilakukan oversampling akan lebih baik menggunakan evaluasi model untuk class imbalance (AUC/F-1), dalam kasus ini kami menganggap kedua feature sama pentingnya, kami ingin menekan tingkat charge off namun kami juga tidak ingin salah memprediksi sehingga kehilangan terlalu banyak nasabah, karena dua hal ini sama penting maka kami akan fokus kepada nilai AUC untuk mengevaluasi model.

Melakukan percobaan dengan 4 Algoritma

**Decision Tree**
AUC: 92.763 (6.640)
**Random Forest**
AUC: 96.248 (0.972)
**Adaboost**
AUC: 95.312 (0.775)
**XGboost**
AUC: 96.429 (0.648)

**Dilihat dari nilai AUC-nya model XGBoost merupakan yang paling baik dengan gap antara AUC train dan test yang sangat kecil sehingga kami memutuskan model yang paling baik untuk memprediksi target adalah XGBoost.**

**Feature Importance**
![image](https://user-images.githubusercontent.com/114790120/195608072-7b1e9731-311b-43ea-ae09-fe0936965ecf.png)

## Insight Dari Fitur-fitur terpenting

**Term**
![image](https://user-images.githubusercontent.com/114790120/195608532-b54af48a-eede-4f57-8241-c95aa22cce8b.png)

Semakin lama waktu pinjaman tingkat charge offnya semakin rendah, hal ini dikarenakan semakin lama jangka waktu pinjaman maka cicilan yang akan dibayarkan setiap bulan akan lebih sedikit sehingga meringankan kreditur dalam membayar pinjamannya dan akan mengurangi kemungkinan gagal bayar. Apabila kreditur terdeteksi charge off salah satu solusi yang dapat ditawarkan adalah dengan menawarkan jangka waktu pinjamannya menjadi lebih lama agar dapat meringankan kreditur, selain itu jangka waktu yang lebih lama juga akan lebih menguntungkan untuk bank karena jumlah revenue dari bunga yang diperoleh akan lebih tinggi.


**UrbanRural**

Reason : Profesor Sekolah Bisnis Universitas Kansas Bob DeYoung menyebutkan bank  selalu mengandalkan hard and soft information dalam memutuskan kelayakan pinjaman. Tetapi di masyarakat pedesaan, hard information tentang kelayakan pinjaman bisa jadi sulit didapat. Misalnya, usaha kecil pedesaan cenderung tidak memiliki laporan keuangan yang diaudit yang mengurangi jumlah hard information tentang kelayakan kredit mereka. Namun, merutnya komunitas pedesaan lebih erat daripada komunitas perkotaan, dan jaringan informasi pribadi ini meluas ke komunitas bisnis pedesaan juga. Ini memberi bank pedesaan sumbangan soft information tentang bisnis dan pengusaha lokal. Bank jelas menggunakan informasi ini dengan baik, karena dibuktikan dengan tingkat default yang rendah pada pinjaman mereka.

Recommendation : 
- Fokus bisnis bisa diperluas untuk daerah Desa, dengan alasan dapat menambah creditur dan pemasukan, disamping itu dapat me-reduce cost dikarenakan fokus panawaran bisnis untuk daerah yang potensi CHGOFF nya rendah.
- Untuk darah Kota, dapat dibuat beberapa policy yang lebih ketat dengan tujuan meminimalisir angka CHGOFF dari kreditur yang berlokasi di Kota, seperti persyaratan dokumen bisnis yang lebih lengkap.

Source : https://today.ku.edu/2012/11/06/loan-default-rates-lower-rural-communities

**StateSame**
Reason : Lokasi bank dan kreditur yang sama memiliki tingkat charge off yang lebih rendah, melayani pinjaman untuk bisnis di negara bagian lain akan lebih sulit karena proses pemantauan bisnis dan proses mendapatkan informasi-informasi untuk menentukan kelayakannya pun akan menjadi lebih sulit.

Recommendation

SBA dapat merekomendasikan bank berdasarkan state yang sama kepada calon kreditur untuk menurunkan kemungkinan charge off.
Memperbaiki policy khusus untuk peminjam yang mengajukan pinjaman pada bank dengan negara bagian yang berbeda.
Bank dapat berforkus pada calon kreditur yang berlokasi sama dengan bank dalam melakukan promosinya, sehingga mereduce cost untuk pemasaran.

**SBA Approved Percentage**
Sebagian pinjaman yang masuk merupakan pinjaman yang mendapatkan jaminan 50% dari SBA yang termasuk ke dalam program SBA Express. Pada programini keputusan kelayakan dibuat oleh pemberi pinjaman (Bank) bukan SBA sehingga pinjaman ini tidak memerlukan dokumen-dokumen yang disyaratkan SBA, pinjaman akan diproses dalam 36 jam. Pinjaman yang masuk kategori ini juga memiliki tingkat charge off yang tinggi. Asumsi kami adalah bahwa sebagian besar bisnis memilih untuk menggunakan program SBA Express karena lebih mudah untuk mendapatkan persetujuan pinjaman dalam waktu yang lebih singkat, sayangnya dalam proses persetujuan ini penilaian kelayakan yang dilakukan bank dalam waktu yang singkat terhadap bisnis kurang baik sehingga banyaknya pinjaman yang gagal bayar, khusus untuk program SBA Express yang membutuhkan waktu cepat model akan sangat membantu dalam memprediksi kelayakan bisnis.


