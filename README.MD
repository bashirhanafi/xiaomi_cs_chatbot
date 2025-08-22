# Xiaomi AI Customer Service Chatbot
![Flow](img/flow.png)

## Overview

### Algoritma 
1. Saat chatbot pertama kali dijalankan, pengguna dapat memberikan pesan sebagai input.
2. Setelah menerima input pengguna, sistem akan memprosesnya melalui API.
3. Lalu API akan menghubungkan ke database.
4. Setelah menghubungkan melalui database, model akan diinisialisasi: Llama3.2:3b, sentence-transformers, dan LangChain sebagai framework.
5. Setelah diinisialisasi, model akan dipadukan dengan memory menggunakan ConversationBufferWindowMemory agar konteks percakapan tetap terjaga.
6. Agent akan memanggil AgentExecutor untuk mengeksekusi input pengguna, hal ini termasuk: model, tools call, prompt (system prompt), dan memory.
7. Jika diperlukan tools, AgentExecutor akan memanggil tools khusus seperti CheckListProduct, CheckOrderStatus, CheckSpecification, atau CheckGuarantee. Namun jika tidak, AgentExecutor akan meneruskannya ke generasi teks.
8. Tools kemudian akan mengambil data (retrieve) dari dokumen atau query data dari database untuk memperkaya konteks.
9. Setelah dilakukan retrieve atau query, model akan dilakukan augmented, yaitu penggabungan LLM + konteks hasil retrieve.
10. Setelah itu, maka akan dihasilkan hasil generasi teks.
11. Hasil history percakapan (input pengguna & hasil generasi teks) akan disimpan melalui database struktural.
12. Hasil generasi teks ini akan menjadi respon pengguna yang diterima melalui format JSON.

## Cara Instalasi & Requirement (Local)
#### 1. Buka terminal dan masuk ke dir project

Example:
```json
cd models
```

#### 2. Instalasi model (asumsi sudah memiliki Ollama)

Example:
```json
ollama run llama3.2:3b
```

#### 3. Instalasi requirement

Masuk ke dir chatbot
```json
cd chatbot
```

Install package yang dibutuhkan
```json
pip install -r requirements.txt
```

#### 4. Setup Database

Buat file .env

```json
DB_USER = "<username>"
DB_PASSWORD = "<password>"
DB_HOST = "<host>"
DB_PORT = "<port>"
DB_NAME = "cs_chatbot"
```

## Desain Database
![Database Design](img/database-design.png)

#### Tabel Customer
| Column Name   | Type         | Description               |
|---------------|--------------|---------------------------|
| id            | INTEGER (PK) | Unique ID of the customer |
| customer_name | VARCHAR      | Full name of the customer |
| email         | VARCHAR      | Email of the customer     |

#### Tabel Order
| Column Name | Type         | Description              |
|-------------|--------------|--------------------------|
| id          | INTEGER (PK) | Unique ID of the order   |
| order_date  | DATE         | Date for the order       |
| customer_id | INTEGER (FK) | REFERENCES customers(id) |
| product_id  | INTEGER (FK) | REFERENCES products(id)  |
| qty         | INTEGER      | Qty for the order        |
| total       | INTEGER      | Total for the order      |
| status      | VARCHAR      | Status for the order     |

#### Tabel Product
| Column Name   | Type         | Description                            |
|---------------|--------------|----------------------------------------|
| id            | INTEGER (PK) | Unique ID of the product               |
| product_name  | VARCHAR      | Name of the product                    |
| specification | VARCHAR      | Directory of the product specification |
| variant       | VARCHAR      | Variant of the product                 |
| qty           | INTEGER      | Qty of the product                     |
| price         | BIG INT      | Price of the product                   |

#### Chat history table
| Column Name | Type         | Description                     |
|-------------|--------------|---------------------------------|
| id          | INTEGER (PK) | Unique ID of the chat history   |
| session_id  | UUID         | Session of the chat history     |
| message     | JSONB        | Message of the chat history     |
| create_at   | TIMESTAMP    | Create time of the chat history |

## List Library & Framework
1. **Python**: digunakan sebagai bahasa pemrograman utama.
2. **LangChain**: Framework untuk menghubungkan LLM dengan data, memory, dan tools eksternal.
3. **Ollama**: Runtime untuk menjalankan model LLM secara lokal.
4. **Model Llama3.2:3b**: Model LLM ringan dari Meta untuk inferensi lokal dengan jumlah parameter 3b.
5. **PostgreSQL**: Database relasional open-source untuk menyimpan data dan chat history.
6. **sentence-transformers**: Library untuk menghasilkan embedding untuk semantic search.
7. **PyPDF2**: Library Python untuk membaca dan memproses file PDF
8. **FastAPI**: Framework modern untuk membangun API dengan dokumentasi otomatis.

## Model LLM
### 1. Llama 3.2
Model Llama 3.2 3B adalah model bahasa besar multibahasa (LLM) yang telah dilatih dan disesuaikan dengan instruksi menggunakan 3 juta parameter. Secara performa, 3B melampaui performa dari model open-source seperti Gemma 2 2.6B dan Phi 3.5 mini model dalam task: following instruction, summarization, tool use, prompt rewriting.

### 2. Sentence Transformers
Sentence Transformers (SBERT) adalah model Transformer yang digunakan untuk mengakses, menggunakan, dan melatih model embedding serta re-ranker. Model ini dapat digunakan untuk menghitung skor similarity menggunakan cross-encoder atau menghasilkan embedding sparse menggunakan sparse-encoder, Model ini dapat digunakan dalam beberapa task: semantic search, semantic textual similarity, paraphrase mining.

## Daftar Pertanyaan
### 1. Sapa Pengguna
#### Request
```json
{
  "message": "Halo, apa kabar??"
}
```
#### Response
```json
{
  "response": "Halo! Kabar baik, saya senang bisa berbicara dengan Anda. Apakah Anda memiliki pertanyaan tentang pesanan atau produk Xiaomi lainnya? Saya di sini untuk membantu Anda."
}
```
### 2. Stok Produk
#### Request
```json
{
  "message": "Stok redmi note 14 apakah sudah ada? Kalo iya, ada variant apa saja?"
}
```

#### Response
```json
{
  "response": "Stok Redmi Note 14 sudah ada dan tersedia dalam beberapa variant. Berikut adalah informasi tentang stok dan harga untuk setiap variant:
  - Mist Purple/8 GB + 128 GB: Stok 10 unit, Harga Rp 2.399.000
  - Mist Purple/8 GB + 256 GB: Stok 3 unit, Harga Rp 2.599.000
  - Lime Green/8 GB + 128 GB: Stok 1 unit, Harga Rp 2.399.000
  - Lime Green/8 GB + 256 GB: Stok 0 unit
  - Midnight Black/8 GB + 128 GB: Stok 12 unit, Harga Rp 2.399.000
  - Midnight Black/8 GB + 256 GB: Stok 13 unit, Harga Rp 2.599.000
  - Sand Gold/8 GB + 128 GB: Stok 2 unit, Harga Rp 2.399.000
  - Sand Gold/8 GB + 256 GB: Stok 0 unit
    Silakan memilih variant yang sesuai dengan kebutuhan Anda dan hubungi kami untuk memesan atau mendapatkan informasi lebih lanjut."
}
```

### 3. Status Pesanan
#### Request
```json
{
  "message": "pesanan atas nama Rizan Rifaldisya ada dimana?"
}
```

#### Response
```json
{
  "response": "Terima kasih telah memilih Xiaomi! Pesanan Anda oleh Rizan Rifaldisya saat ini dalam proses pengiriman dan sedang dalam perjalanan ke alamat Anda. Silakan menunggu pesanan Anda tiba di tempat Anda. Jika Anda memiliki pertanyaan atau kekhawatiran lain, jangan ragu untuk bertanya!"
}
```

### 4. Spesifikasi Produk
#### Request
```json
{
  "message": "Aku mau tahu spesifikasi dari Poco F7 dong"
}
```

#### Response
````json
{
  "response": "Poco F7 memiliki spesifikasi sebagai berikut:
  * Prosesor: Platform Mobile Snapdragon 8s Gen 4
  * CPU: Prosesor octa-core, hingga 3,21 GHz
  * GPU: Adreno 825
  * Al Engine: Qualcomm Hexagon NPU
  * Penyimpanan & RAM: 12 GB + 512 GB (LPDDR5X + UFS 4.1)
  * Dimensi: Tinggi 163,1 mm, Lebar 77,9 mm, Tebal 8,2 mm
  * Layar: 6,83'' 1.5K, Rasio menyegarkan layar hingga 120 Hz, Rasio pengambilan sampel sentuh hingga 480 Hz
  * Kecerahan std.: 700 nit, Kecerahan HBM: 1.700 nit, Kecerahan puncak: 3.200 nit
  * Sensor: Sensor jarak, Sensor cahaya sekitar, Akselerometer, Kompas elektronik, Blaster, Giroskop, Sensor kedip
  * Motor getaran: Motor getaran linear sumbu X
  * Sistem Operasi: Didukung oleh Xiaomi HyperOS 2
  * Isi Kemasan: Ponsel / Adaptor / Kabel USB Tipe-C / Alat Pembuka SIM / Casing Pelind
  * Fitur video kamera depan: Selfie gerak lambat, Selfie selang waktu, Teleprompter video
  
  Semoga informasi ini membantu Anda dalam memilih spesifikasi yang tepat untuk Poco F7."
}
````

### 5. Garansi Produk
#### Request
```json
{
  "message": "Bagaimana garansi produk dari Xiaomi 15? saya ingin tahu detailnya"
}
```
#### Response
```json
{
  "response": "Berikut adalah detail garansi produk dari Xiaomi 15:
  1. Alamat lengkap pelanggan dan kontak yang dapat dihubungi harus disediakan oleh pelanggan.
  2. Nomor Pembelian dan duplikat faktur pembelian pelanggan yang asli harus disertakan dalam permohonan garanti.
  3. Garansi ini tidak mencakup beberapa kasus, seperti:
    * Nomor serial produk, segel garanti hilang atau telah dihapus, dirusak dan diubah.
    * Aksesoris atau bagian luar produk yang hilang.
    * Perawatan umum, update atau pemasangan aplikasi, demo produk, atau layanan yang tidak terkait dengan perbaikan/penggantian.
    * Kerusakan produk yang disebabkan oleh pembongkaran dan penggunaan secara normal, termasuk karat dan noda meskipun tidak terbatas pada hal tersebut.
  4. Garansi ini tidak mencakup beberapa kasus lainnya yang berlawanan dan tidak sesuai dengan etika bisnis.
  5. Xiaomi akan menentukan suatu produk telah “Kehilangan Garanti” dengan kebijakan perusahaan yang sesuai dengan standar di bawah ini.
  
  Perbaikan produk yang telah kehilangan garanti harus dipisahkan dari pusat layanan Xiaomi serta perbaikan yang dilakukan terhadap produk tersebut akan dikenai biaya tambahan.
  
  Selama periode garansi, jika terjadi pelanggaran yang dilakukan oleh pelanggan, seperti perusakan, perbaikan sendiri, terpapar air, kerusakan karena salah penggunaan, penggantian, kesalahan memahami panduan produk, dll., maka permohonan garanti akan ditolak."
}
```



## Daftar Toolcall
### 1. Check List of Product
```json
class OrderListInput(BaseModel):
    product_name: str = Field(..., description="Nama produk untuk cek mengecek ketersediaan produk")
    variant: Optional[str] = Field(..., description='Variant dari produk untuk mengecek ketersediaan produk')
    question: str = Field(..., description="query dari user")

class CheckListProduct(BaseTool):
    name: str = "check_list_product"
    description: str = "Hanya gunakan tool ini jika pengguna ingin mengecek ketersediaan product_name dalam database."
    args_schema: Type[BaseModel] = OrderListInput

    def _run(self, product_name: str, question: str, variant: str = None):
    # implementation
```

### 2. Check Order Status
```json
class OrderStatusInput(BaseModel):
    customer_name: str = Field(..., description="Nama customer untuk cek status pesanan")
    question: str = Field(..., description="query dari user")

class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = "Hanya gunakan tool ini jika pengguna menanyakan status pesanan tertentu berdasarkan customer_name."
    args_schema: Type[BaseModel] = OrderStatusInput 

    def _run(self, customer_name: str, question: str):
    # implementation
```

### 3. Check Specification
```json
class CheckSpecificationInput(BaseModel):
    product_name: str = Field(..., description="Nama produk untuk cek spesifikasi produk")
    question: str = Field(..., description="query dari user")

class CheckSpecification(BaseTool):
    name: str = "check_specification"
    description: str = "Hanya gunakan tool ini jika pengguna menanyakan tentang spesifikasi atau keunggulan produk."
    args_schema: Type[BaseModel] = CheckSpecificationInput 

    embeddings: Any = Field(default=None)

    def _run(self, product_name: str, question: str):
    # implementation
```

#### 4. Check Guarantee
```json
class CheckGuaranteeInput(BaseModel):
    product_name: str = Field(..., description="Nama produk untuk cek garansi produk")
    question: str = Field(..., description="query dari user")

class CheckGuarantee(BaseTool):
    name: str = "check_guarantee"
    description: str = "Hanya gunakan tool ini jika pengguna menanyakan tentang garansi produk"
    args_schema: Type[BaseModel] = CheckGuaranteeInput 

    embeddings: Any = Field(default=None)

    def _run(self, product_name: str, question: str):
    # implementation
```

## Cara run FastAPI (local)
#### 1. Pastikan fastapi & uvicorn sudah terinstall

Example:
```json
pip install fastapi uvicorn
```

#### 2. Jalan menggunakan uvicorn

Example:
```json
uvicorn main:app --reload
```

#### 3. Jalankan di Browser

Example:
```json
http://127.0.0.1:8000/docs
```

