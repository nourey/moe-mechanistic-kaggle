# OLMoE Kaggle ortam tanılaması

## Güncel adım: gerçek OLMoE yükleme ve koşullu küçük EP2 inference — `olmoe-v0.6.0`

`moe-storage-20261122T074215Z-763f1ffd.zip` doğrulandı: ZIP CRC, 3/3 içerik özeti ve 4/4 kaynak özeti eşleşiyor. ZIP SHA-256: `9c766032a07c8bb0829e1edf1812ead62783e5d0c6b8446c703efb56487fda58`. Yeni oturumda eski venv yok. `/kaggle/working` yaklaşık 19,5 GiB boş; `/kaggle/temp` için ölçülen üst dizin farklı bir dosya sisteminde ve küçük yazma testi geçti. Bildirilen TB boş alan kullanıcı kotası kanıtı değildir. RAM cgroup limiti 30 GiB. Aşağıdaki sürüm yeni oturum içindir; eski A/B/C komutlarını veya Run All'ı tekrar kullanma.

**Önce Kaggle'da GPU T4 ×2 ve Internet On seç. Sonra yalnız aşağıdaki hücreyi çalıştır.** Ayar değişikliği yeni oturum açarsa sorun değil; eski venv veya ZIP gerekmiyor.

```python
import subprocess, sys, json
from pathlib import Path
from IPython.display import display, FileLink
repo = '/kaggle/working/moe-mechanistic-kaggle-olmoe-v0.6.0'
if not Path(repo).exists():
    subprocess.run(['git', 'clone', '--branch', 'olmoe-v0.6.0', '--depth', '1',
        'https://github.com/nourey/moe-mechanistic-kaggle.git', repo], check=True)
subprocess.run(['git', '-C', repo, 'describe', '--exact-match', '--tags', 'HEAD'], check=True)
result = subprocess.Popen([sys.executable, '-u', '-B', repo + '/model_qualify.py'],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
receipt = None
for line in result.stdout:
    print(line, end='', flush=True)
    if line.startswith('EVIDENCE_JSON:'):
        receipt = json.loads(line.split(':', 1)[1])
result.wait()
if receipt:
    display(FileLink(receipt['archive']))
else:
    raise RuntimeError('Kanıt ZIP kaydı yok; yukarıdaki hatayı gönder.')
```

İndirilecek **`moe-model-….zip`** dosyasını geri gönder. Kurulum/model indirme dahil toplam süre üst sınırı 40 dakika; her aşama adı anında görünür. Hata olursa otomatik sürüm değiştirme veya ikinci kurulum denemesi yok. Yalnız tamamlanmış ve aynı lock ile doğrulanmış kendi venv'ini tekrar kullanabilir; eksik/başka venv'i üzerine yazmaz. Özgün ZIP'ler ve diğer ortamlar silinmez.

Bu adım yaklaşık 9 GB izole runtime ile 13.841.130.907 bayt sabitlenmiş checkpoint/tokenizer indirir. Aynı 20 GB diske birlikte sığmaları varsayılmaz: model **`/kaggle/temp/moe-mechanistic-1525`** altına, venv `/kaggle/working` altına yazılır. Farklı filesystem, canlı boş alan ve yazma kontrolü zorunlu; gerçek kota/indirme hatasında durur. Model ve venv kanıt ZIP'ine dahil edilmez. Ana Python ortamı değiştirilmez; seçili ana paket sürümleri önce/sonra karşılaştırılır.

`runtime-lock.txt`, önceki kurulum raporundaki 199 wheel'in sürüm ve SHA-256'larını dondurur. Yalnız Linux x86_64 / Python 3.13 kabul edilir. Yeni venv, pip seeding ve kurulum yalnız resmi PyPI binary artifact'leriyle yapılır. `pip check`, wheel raporu, gerçek CUDA ordinal–GPU UUID eşlemesi ve native importlar yeniden kontrol edilir; sentetik B/C deneyleri tekrarlanmaz. Önceki nonfatal `sitecustomize / wrapt` uyarısı otomatik ana ortam değişikliği gerekçesi değildir.

### Sınırlı kabul ve kanıtın anlamı

* **D:** Native `AsyncLLM` ile TP1/PP1/DP2/EP2; gerçek OLMoE revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`, FP16, Triton MoE, `allgather_reducescatter`. Tek süreçli `LLM(DP2)` bu sürümde desteklenmediğinden kullanılmaz. Şardlar indirme sonrası streaming hash ile doğrulanır; LFS pointer Git hash'i tensor içeriğinin hash'i sayılmaz.
* Her worker'ın mevcut named extension/RPC arayüzü yalnız okuma için kullanılır: GPU UUID/PID, native DP/EP grupları, 16 katmandaki gerçek global→local expert haritaları ve GPU KV kapasitesi. İki rank'te 32+32 sayısını toplamak yeterli değildir; her global expert'in tam bir rank sahibi olmalıdır. GPU KV kapasitesi request allocation/ownership ledger değildir. Worker PID, EngineCore incarnation nonce değildir.
* **Koşullu küçük E functional smoke:** D geçerse iki rank-hint'li istek, her biri en fazla 8 token. Native terminal kayıtları ve worker PID/device ile eşlenen CUDA MoE/AllGather/ReduceScatter izleri saklanır. Genel NCCL koordinasyon kernel'i payload collective kanıtı sayılmaz. Eksik profiler eşlemesi veya farklı/variable-size payload yolu `PARTIAL` kalır. Explicit DP hint + kaynak seçici, bağımsız runtime request→engine binding receipt'i yerine sunulmaz. Tam E scheduler/KV/request-ownership qualification henüz kapanmaz.
* Model kaynakları değiştirilmez. `VLLM_ALLOW_INSECURE_SERIALIZATION=0`; arbitrary pickled RPC callback kullanılmaz. Worker extension constructor, inference override, network veya background loop içermez. Eager/512-token/four-sequence limitleri functional test sınırlarıdır; performans ayarı, araştırma SLO'su veya kalibrasyon değildir.
* DBO/EPLB/prefix-cache kapalı. Warmup ve JIT süreleri ölçüm amaçlı değildir. Normal shutdown dönüşü kaydedilir; timeout/kill veya dönüş tek başına fiziksel graceful-drain kanıtı sayılmaz. OLMoE FP16/AGRS, Qwen BF16/DeepEP veya dört/altı-GPU Q kabulü değildir. M1-002/G0 durumları değiştirilmez.

Aşağıdaki bölümler önceki sürümlerin tarihsel talimatlarıdır; güncel oturumda bunları tekrar çalıştırma.

## İletişim C PASS — model için kalan ilk önkoşul: depolama

`moe-comm-20261121T221312Z-3bde775e.zip` doğrulandı: CRC, 25/25 dosya ve 17/17 kaynak özeti eşleşiyor. Ham FP16 NPY dosyalarından her rank'te 8.192 AllGather ve 4.096 ReduceScatter değeri yeniden kontrol edildi; uzak veri, sıra ve destination rank sahipliği doğru. CUDA izlerinde her cihazda gerçek `ncclDevKernel_AllGather_RING_LL` ve `ncclDevKernel_ReduceScatter_Sum_f16_RING_LL` var. Mevcut vLLM PyNccl API dosyaları pinned kaynakla eşleşiyor. NCCL `2.29.7+cuda13.2` raporlanmış; PyTorch build CUDA `13.0` etiketi ayrı bir gözlemdir. Bu koşulda iki primitive başarıyla çalıştı; tüm CUDA stack kombinasyonlarına genellenmez.

Native loglarda iki yönde `via SHM/direct/direct` seçilmiş. `NET/Socket` ve P2P-capability satırları bu seçilmiş payload kanalının yerine geçmez. GPU tamponları/NCCL CUDA yürütmesi doğrulandı; doğrudan GPU P2P, NVLink veya RDMA iddiası yok. Her rank'in Init/payload/Abort kayıtları kendi communicator pointer'ıyla bağlı; ortak commId ve fixture'ın iki üyesiyle birlikte kontrol edildi. Her iki owned communicator için `Abort COMPLETE` gözlendi ve süreçler join oldu. Orijinal runtime receipt'lerindeki `native_abort_completion_verified=false` değiştirilmedi; bu ek bulgu offline native log incelemesidir, serving graceful-drain kanıtı değildir.

A/B/C sınırlı qualification adımları geçti. Tam OLMoE yükleme, vLLM `AgRsAll2AllManager`, ragged/idle rank popülasyonları, EP2 serving, scheduler/KV ve BLIS kalibrasyonu henüz yapılmadı.

[Resmi Hugging Face kaydından](https://huggingface.co/allenai/OLMoE-1B-7B-0924/tree/6d84c48581ece794365f2b8e9cfb043c68ade9c5) model revision'ı `6d84c48581ece794365f2b8e9cfb043c68ade9c5` olarak sabitlendi. `model-pin.json`; üç BF16 safetensors shard'ının LFS SHA-256'larını, dosya boyutlarını ve doğrulanmış küçük config/index kimliklerini içerir. Runtime hedefi FP16 olacak; checkpoint BF16 olması T4'te BF16 çalıştırma talimatı değildir. Ağırlıklar indirilmedi.

Üç ağırlık dosyası **13.838.721.960 bayt**, seçilen tüm model/tokenizer dosyaları **13.841.130.907 bayt**. Son `/kaggle/working` boş alanı yaklaşık 12 GB idi. Bu nedenle önce explicit cache alanını seçmek gerekiyor. Host dosya sisteminin TB seviyesinde boş alan raporlamasını kullanıcı/container kotası olarak kabul etmiyoruz.

**`storage-v0.5.0` yalnız bu eksik ortam bilgisini toplar:** `/kaggle/working`, `/kaggle/temp`, HF cache yollarının mevcut üst dizinleri, boş alan, filesystem device ID, küçük geçici yaz/oku kontrolü ve cgroup RAM limitleri. Kendi küçük geçici dosyası kapanınca kaldırılır; mevcut dosyalar silinmez. Eksik hedef dizin için üst dizin ölçüldüğü açıkça belirtilir. Model/paket indirmez, GPU kullanmaz, Model D PASS üretmez. Sana depolama ayarlarını yorumlama işi düşmez; ZIP'i geri gönder.

Mevcut notebook'un altına yeni Code hücresi ekleyip **yalnız bunu** çalıştır:

```python
import subprocess, sys, json
from IPython.display import display, FileLink
repo = '/kaggle/working/moe-mechanistic-kaggle-storage-v0.5.0'
subprocess.run(['git', 'clone', '--branch', 'storage-v0.5.0', '--depth', '1',
    'https://github.com/nourey/moe-mechanistic-kaggle.git', repo], check=True)
result = subprocess.run([sys.executable, '-B', repo + '/model_storage.py'], text=True, capture_output=True)
print(result.stdout)
print(result.stderr)
result.check_returncode()
receipt = json.loads(next(s.split(':', 1)[1] for s in reversed(result.stdout.splitlines()) if s.startswith('EVIDENCE_JSON:')))
display(FileLink(receipt['archive']))
```

`moe-storage-….zip` dosyasını gönder. Bu soru gerçek model indirmesinden önce gereklidir; A/B/C testleri tekrarlanmaz. Kaggle oturumunu veya accelerator ayarını değiştirme; mevcut sanal ortamı koruyalım. Tam GPU model yükleme komutu, cache yolu ve sınırları belirlenince hazırlanacak.

## B PASS — sıradaki C: iki-GPU native NCCL iletişimi

`moe-kernel-20261121T214329Z-7706125c.zip` bağımsız incelendi: CRC, 17/17 dosya özeti ve 14/14 kaynak özeti eşleşiyor. Ham NPY dosyalarından her GPU'daki 16.384 değer yeniden karşılaştırıldı: en büyük hata `0.000490427`, sınır `0.02`, uyuşmazlık/NaN yok. Sıfır çıktı negatif kontrolü reddediliyor. CUDA izleri her cihazda doğru device ID'siyle iki `fused_moe_kernel`, bir aktivasyon ve bir MoE sum gösteriyor. İki giriş/routing özeti aynı, bu tek çiftte çıktılar bit düzeyinde aynı. Kurulu `fused_moe.py` dosyası incelenen pinned kaynak dosyasıyla byte/hash olarak eşleşiyor; bu tüm vLLM build'inin aynı commit veya producer patch olduğu iddiası değildir.

B yalnız sentetik yerel FP16 MoE için PASS. Tam OLMoE, EP2 expert paylaşımı, scheduler/KV ve serving hâlâ test edilmedi. Eksik T4 tuning dosyası nedeniyle varsayılan kernel launch konfigürasyonu kullanılmış; bu CPU fallback değildir, ancak zamanlar performans/kalibrasyon kanıtı değildir. `wrapt` uyarısı ham loglarda korunur.

**C sürümü: `nccl-v0.4.0`.** Aynı sanal ortamda iki yeni geçici process, vLLM'in değişmemiş `PyNcclCommunicator` bileşenini kullanır. Gloo/FileStore yalnız CPU bootstrap/kimlik alışverişidir; veri payload'ı GPU FP16 tamponlarıyla native NCCL AllGather ve ReduceScatter SUM üzerinden gider. Sürüm/kaynak/cihaz uyuşmazlığı kapalı kalır; farklı kütüphane, paket veya transport ayarı denenmez. Constructor'ın mevcut bir-eleman warmup AllReduce'u ayrıca gerçekleşir.

Her rank'in farklı payload'ı var. Toplamların eşleşmesi yeterli değil: uzak rank verisinin doğru sırada gelmesi ve ReduceScatter sonucunun doğru rank'e dönmesi bit düzeyinde kontrol edilir. NaN ile doldurulan hedefler, yazılmayan veya eksik çıktıyı reddeder. Her rank'te native NCCL CUDA olayları ve tam debug logları tutulur. Uygulama CPU/Gloo payload collective'ine geçmez; NCCL'nin kendi iç taşıması SHM/host staging kullanabilir. Fiziksel P2P/SHM/NET yolu backend adından çıkarılmaz. Eşit boyutlu primitive testi; `AgRsAll2AllManager`, değişken/idle token popülasyonları ve tam EP2 serving doğrulanmaz. `PyNccl.destroy()` dönüşü ve child process join'i kaydedilir; wrapper'ın timeout'lu abort thread'inin tamamlandığı bağımsız olarak iddia edilmez. Gloo timeout 60s, tüm coordinator grubu için sınır 180s; tekrar yok.

**Mevcut Kaggle notebook'una yeni Code hücresi ekle; yalnız bunu çalıştır, Run All yapma:**

```python
import subprocess, sys, json
from IPython.display import display, FileLink
repo = '/kaggle/working/moe-mechanistic-kaggle-nccl-v0.4.0'
subprocess.run(['git', 'clone', '--branch', 'nccl-v0.4.0', '--depth', '1',
    'https://github.com/nourey/moe-mechanistic-kaggle.git', repo], check=True)
result = subprocess.run([sys.executable, '-B', repo + '/nccl_check.py',
    '--prior-archive', '/kaggle/working/moe-kernel-20261121T214329Z-7706125c.zip',
    '--prior-sha256', '0dfecc0f95b55864714f26a2044cbfd0833e4a1ee919e41c9165e9e7bcf7ea47'],
    text=True, capture_output=True)
print(result.stdout)
print(result.stderr)
result.check_returncode()
receipt = json.loads(next(s.split(':', 1)[1] for s in reversed(result.stdout.splitlines()) if s.startswith('EVIDENCE_JSON:')))
display(FileLink(receipt['archive']))
```

Üretilen `moe-comm-….zip` dosyasını gönder. Model/paket indirilmez, A/B tekrar çalıştırılmaz; önceki oturum yoksa kurulum yapmadan durur. Marker dizileri ve `.npy` çıktılar, native NCCL sürümü/kütüphane adı, actual kaynak özetleri, CUDA izleri, rank stdout/stderr ve önceki B ZIP'i korunur. Tampon baytları uygulama düzeyi boyutlardır; fiziksel link baytları veya bant genişliği değildir. D/E/M ve BLIS kalibrasyonu çalıştırılmaz.

## Önceki A sonucu ve B hazırlığı

`moe-runtime-20261121T212403Z-5a9e8e9a.zip` doğrulandı: CRC, 92/92 iç dosya özeti, 11/11 runner özeti ve önceki başarısız ZIP'in değişmeden taşındığı doğrulandı. İki native CUDA uzantısı yükleniyor; PyTorch `2.13.0+cu130`, vLLM `0.31.0`, Triton `3.7.1`, Transformers `5.17.0`. Yeniden kurulum yok, ana ortamın seçili sürümleri değişmemiş. `wrapt` / `sitecustomize` mesajı hâlâ var; başarılı native komutun stderr'inde korunuyor. Bu mesajı gidermek için paket eklenmedi.

**B sürümü: `moe-kernel-v0.3.0`.** Aynı iki T4'te sırayla birer gerçek `vllm...fused_moe.fused_experts` çağrısı yapar. Yeni paket veya model indirmez. Normal Triton ilk-çağrı JIT derlemesi olabilir; bağımlılık/toolchain derlemesi veya alternatif backend denemesi yapılmaz.

Fixture sentetik FP16 ağırlıklar, 8 token, 64 uzman, top-8, hidden 2048 / intermediate 1024 ve SiLU-gated MLP kullanır. Her uzman seçilir; routing CPU'da bilinir ve kaydedilir. Pretrained OLMoE ağırlıkları, gerçek model router'ı, tam serving backend seçimi veya model residency test edilmez. Aynı girdiler iki cihazda ayrı ayrı yürütülür; bu DP2/EP2, expert sharding veya GPU'lar arası collective kanıtı değildir.

Kabul: çıktı seçilen GPU'da FP16 olmalı; bağımsız FP32 CPU formülüyle `atol=0.02, rtol=0` karşılaştırması geçmeli. Tolerans pinned vLLM kernel testinden alınan mühendislik kontrolüdür, araştırma SLO'su değildir. Sıfır çıktı negatif kontrolü reddedilmeli; CUDA profiler'da en az iki `fused_moe_kernel` olayı bulunmalı. CPU etiketleri CUDA olayı sayılmaz. Matematik geçip CUDA izi yoksa PARTIAL, ilk başarısız cihazda durur. CUDA/CUPTI profili kurulamazsa hata korunur; bu tek başına MoE kernel uyumsuzluğu kanıtı sayılmaz. Zamanlar ilk JIT ve profiling içerir, kalibrasyon/performans verisi değildir.

**Mevcut Kaggle notebook'unun altına yeni Code hücresi ekle; yalnız bunu çalıştır, Run All yapma:**

```python
import subprocess, sys, json
from IPython.display import display, FileLink
repo = '/kaggle/working/moe-mechanistic-kaggle-kernel-v0.3.0'
subprocess.run(['git', 'clone', '--branch', 'moe-kernel-v0.3.0', '--depth', '1',
    'https://github.com/nourey/moe-mechanistic-kaggle.git', repo], check=True)
result = subprocess.run([sys.executable, '-B', repo + '/moe_kernel.py',
    '--prior-archive', '/kaggle/working/moe-runtime-20261121T212403Z-5a9e8e9a.zip',
    '--prior-sha256', 'b37a822e0a89fdf7411c0d06d4c6b779064f267d97460afd8f44caa813699e92'],
    text=True, capture_output=True)
print(result.stdout)
print(result.stderr)
result.check_returncode()
receipt = json.loads(next(s.split(':', 1)[1] for s in reversed(result.stdout.splitlines()) if s.startswith('EVIDENCE_JSON:')))
display(FileLink(receipt['archive']))
```

Yeni `moe-kernel-….zip` dosyasını gönder. Önceki Kaggle oturumu/venv yoksa kurulum yapmadan BLOCKED olur. ZIP; önceki A kanıtını, exact runner commit/kaynak özetlerini, girdi tensor özetleri/routing'i, tam stdout/stderr'i, sayısal sonuçları (`.npy`, pickle yok), CUDA izini ve her cihazın sonucunu içerir. A tanılaması ve konfigürasyonu kendi kapsamıyla korunur; B'nin kapsamı ayrıca `kernel.json` içindedir. Sonraki C/D/E/M bu sürümde çalıştırılmaz.

## İlk GPU ortam sonucu ve sıradaki dar düzeltme

21 Kasım 2026 tarihli `env-v0.1.0` kanıtı doğrulandı: arşiv CRC'si, 31/31 iç dosya özeti ve altı kaynak özeti yayımlanan `c1783f1c5fbf310290e17ffc5a7ce729e926fee4` commit'iyle eşleşiyor. İki gerçek T4/SM75 görülüyor; PyTorch CUDA cihaz sorguları çalışıyor. A aşaması yalnız `ModuleNotFoundError: No module named 'vllm'` nedeniyle FAIL. MoE kernel, gerçek NCCL aktarımı ve model-serving çalıştırılmadı.

Güncel Kaggle PyTorch'u `2.11.0+cu130`, Transformers `5.20.2`. [vLLM 0.31.0'ın resmi paket metadatası](https://pypi.org/pypi/vllm/0.31.0/json) PyTorch `2.13.0` ve Transformers `>=5.10.4,<5.18.0` istiyor. Bu fark kurulmuş bir ABI hatası değildir; henüz vLLM kurulmamış.

**Yeni ortam için notebook: `Kaggle_Isolated_Runtime.ipynb` (`runtime-v0.2.1`).** Yeni private notebook'ta T4 ×2 / Internet On ile Run All kullan. Kaggle ana ortamını değiştirmeden ayrı, yeni bir venv oluşturur; ensurepip kullanmaz. Resmi PyPI'den yalnız hazır ikili paketlerle tek kurulum dener; derleme, alternatif sürüm veya otomatik ikinci deneme yapmaz. Bu birkaç GB indirme/depolama gerektirebilir. Hata olursa ZIP üretip durur. Mevcut ortamları veya önceki kanıtları silmez. **v0.2.0 ortamın hâlâ açıksa yeniden kurma; aşağıdaki yeniden kontrol hücresini kullan.**

Bu adım yalnız kurulum, `pip check`, `vllm._C_stable_libtorch` / `vllm._moe_C_stable_libtorch` native importları ve izole A tanılamasıdır. Model indirmez, MoE kernel veya iki-GPU collective çalıştırmaz. Başarılı olsa bile B/C/D/E/M aşamaları çalıştırılmamış kalır. Ana ortamın seçili paket sürümlerini önce/sonra karşılaştırır; tüm dosya sisteminin değişmediğini iddia etmez.

Bana yalnız yeni `moe-runtime-....zip` dosyasını gönder. İçinde tam kurulum stdout/stderr, komutlar, PyPI wheel URL/sürümleri/SHA-256 içeren pip raporu, runner kaynak özetleri, önce/sonra tanılamaları ve sonuç bulunur. vLLM wheel'inin adı ve raporlanan SHA-256'sı incelenen resmi sürümle karşılaştırılır. Bu yayın wheel'inin kabulü, eski kaynak checkout'undaki EngineCore producer patch'inin kurulduğu veya birebir kaynak commit'inin doğrulandığı anlamına gelmez. Transitive paketler pip raporunda kaydedilir; henüz dondurulmuş tam bir lockfile yoktur.

Kaggle'ın sistemde görünen CUDA/NCCL paket metadatası ile PyTorch'un bildirdiği yüklü runtime sürümleri farklı olabilir. İlk kanıt, gerçek yüklenen dosyaların tamamını belirlemediği için bu fark tek başına çakışma olarak yorumlanmıyor. İzole import sonucu görülmeden CUDA veya T4 uyumsuzluğu kabul edilmeyecek.

## 22 Kasım: v0.2.0 sonucu ve kurulum yapmadan yeniden kontrol

`moe-runtime-20261121T210750Z-1d519346.zip` incelendi: CRC, 60/60 iç dosya özeti ve 11/11 runner kaynak özeti eşleşiyor. İkili paket kurulumu ve `pip check` geçmiş; ana ortamın seçili sürümleri değişmemiş. v0.2.0 kontrolümüzde eski `vllm._C` adını kullanmamız nedeniyle native-import adımı durdu. Pinned vLLM `db9527a...` kaynağının `CMakeLists.txt:367` ve `vllm/platforms/cuda.py:237` bölümleri CUDA için stable-libtorch modüllerini tanımlıyor. Bu doğrulama hatası v0.2.1'de düzeltildi; düzeltilmiş modüllerin gerçek Kaggle import sonucu henüz elde edilmedi.

`sitecustomize` / eksik `wrapt` mesajı başarılı kurulum sırasında da görülmüş. Önceki `ModuleNotFoundError: No module named 'vllm._C'` sonucunun nedeni bu uyarı değil. Bu düzeltmede `wrapt`, CUDA veya başka bağımlılık eklenmiyor/değiştirilmiyor.

**Aynı Kaggle oturumunda**, mevcut notebook'un en altına tek hücre ekleyip yalnız bu hücreyi çalıştır. Run All yapma. Hücre küçük runner deposunu alır; pip/venv kurulum komutları çalıştırmaz. Eski ZIP'in SHA-256'sını ve iç özetlerini doğrular; eski sanal ortam bulunamazsa kurulum yapmadan durur. Önceki ZIP yeni kanıta değiştirilmeden eklenir.

```python
from pathlib import Path
import subprocess, sys, json
from IPython.display import display, FileLink
repo = Path('/kaggle/working/moe-mechanistic-kaggle-runtime-v0.2.1')
if repo.exists():
    raise RuntimeError('Bu runner klasörü zaten var; önceki kanıtı koru ve dur.')
subprocess.run(['git', 'clone', '--branch', 'runtime-v0.2.1', '--depth', '1',
    'https://github.com/nourey/moe-mechanistic-kaggle.git', str(repo)], check=True)
cmd = [sys.executable, '-B', str(repo / 'prepare_runtime.py'),
    '--resume-archive', '/kaggle/working/moe-runtime-20261121T210750Z-1d519346.zip',
    '--resume-sha256', 'd58dd1416e8fbc20105f82e2b6dd8d418fcb2c3ec7c615ed8445a9ca207e2772']
result = subprocess.run(cmd, text=True, capture_output=True)
print(result.stdout)
print(result.stderr)
result.check_returncode()
receipt = json.loads(next(s.split(':', 1)[1] for s in reversed(result.stdout.splitlines()) if s.startswith('EVIDENCE_JSON:')))
display(FileLink(receipt['archive']))
```

Yeni ZIP'i geri gönder. Burada doğru modülleri import edebilmek MoE yürütme veya EP2 kabulü değildir. Eski `runtime-v0.2.0` etiketi ve asıl başarısız kanıt korunur.

## İlk sürümün kapsamı

Bu ilk sürüm yalnız **A: güncel Kaggle ortamını öğrenme** aşamasını çalıştırır. Paket kurmaz/güncellemez, model indirmez, CUDA tensor/kernel veya NCCL collective çalıştırmaz ve sunucu başlatmaz. PyTorch'un cihaz/peer sorguları sürücü bağlamını başlatabilir; bunlar veri aktarımı veya MoE yürütme testi değildir.

OLMoE-1B-7B-0924 FP16, TP1/DP2/EP2 ve AllGather/ReduceScatter sonraki adaydır. Başarısı henüz gösterilmedi. Qwen BF16/DeepEP, D2 müdahalesi, BLIS kalibrasyonu ve G0 bu tanılamanın kapsamı dışındadır.

## Kaggle'da çalıştırma

1. Yeni, private bir Kaggle notebook aç. Accelerator **GPU T4 ×2**, Internet **On** seç.
2. Bu depodaki `Kaggle_Environment_Diagnostic.ipynb` dosyasını import edip Run All kullan. Notebook `env-v0.1.0` etiketini clone eder ve gerçek commit'i kanıta yazar.
3. Hücrenin verdiği `moe-env-...zip` dosyasını indirip geri gönder. Arayüzde bağlantı çalışmazsa `/kaggle/working` dosya listesinden aynı ZIP'i indir. Çıktıları yorumlaman veya ayrı bağımlılık araştırması yapman gerekmiyor.

Alternatif olarak yeni notebook'ta:

```python
import subprocess, sys
from pathlib import Path
repo = Path('/kaggle/working/moe-mechanistic-kaggle-env-v0.1.0')
if repo.exists():
    raise RuntimeError('Bu çalışma klasörü zaten var. Mevcut çalışmayı silmeden yeni notebook/çalışma klasörü kullan.')
subprocess.run(['git', 'clone', '--branch', 'env-v0.1.0', '--depth', '1',
                'https://github.com/nourey/moe-mechanistic-kaggle.git', str(repo)], check=True)
subprocess.run([sys.executable, '-B', str(repo / 'diagnose.py'),
                '--output-root', '/kaggle/working'], check=True)
```

Etiketin gerçek commit karşılığı teslim mesajında ayrıca verilir. Değişmez bir tekrar için clone sonrasında bildirilen SHA'yı checkout et; uzak HEAD veya etiket adını bir source SHA gibi raporlama. Her çalıştırma yeni klasör/ZIP üretir; önceki kanıtlar üzerine yazılmaz.

## Hangi bilgiyi neden istiyoruz?

| Gerekli bilgi | Neden | Çalıştırılacak işlem / gönderilecek çıktı |
|---|---|---|
| GPU sayısı, UUID, SM, VRAM, sürücü | Gerçek iki-T4 tahsisini ve bellek alanını belirlemek | Aynı notebook/hücre; ZIP içindeki `environment.json`, `nvidia-smi.log` |
| Python/PyTorch/vLLM/Triton ve import hataları | Mevcut paketleri varsaymadan küçük düzeltmeyi seçmek | Aynı işlem; `environment.json`, `logs/torch.*`, `logs/vllm.*`, `logs/triton.*` |
| CUDA/NCCL ve peer/topoloji bilgisi | Sonraki gerçek kernel/iletişim testinin mümkün yolunu seçmek | Aynı işlem; `environment.json`, `logs/topology.*`; henüz collective testi değil |
| Disk ve model-cache ayarı | Model indirmeden depolama sınırını görmek | Aynı işlem; `environment.json` |

**Bana yalnız üretilen ZIP'i gönder.** Notebook ayarlarını yanlışlıkla değiştirmiş olsan da önce kanıtı koru; tanılama sonucu bir sonraki düzeltmeyi belirler.

## Kanıt paketi ve sonuçların anlamı

ZIP; gerçek runner Git HEAD'i, dirty-state, kaynak SHA-256 özetleri, aday konfigürasyonu, komutlar/exit kodları, tam temizlenmiş stdout/stderr, ortam, durum, dosya manifesti ve `SHA256SUMS` içerir. ZIP'in SHA-256 receipt'i ayrıca yanına yazılır. API anahtarları ve auth dosyaları toplanmaz; yalnız seçili cache/CUDA/NCCL ayarları kaydedilir. Tanılama mevcut model-cache içeriğini taramaz.

`collection_status=PASS` ve process exit 0, **kanıt toplamanın tamamlandığı** anlamına gelir. `A_environment=PASS` de yalnız cihaz/import önkoşulları içindir. MoE kernel, iki-GPU veri iletişimi, model yükleme ve gerçek DP2/EP2 serving **UNKNOWN / NOT_EXECUTED** kalır. `nvidia-smi` CUDA başlığı sürücü kabiliyetidir; PyTorch build CUDA ve sorgulanabilirse libcudart sürümü ayrı alanlardır.

T4 SM75 üzerinde Triton/unquantized MoE uygunluğu çözülmüş sayılmaz. Peer access'in false olması NCCL'nin kesin çalışamayacağı anlamına gelmez; seçilen fiziksel taşıma yolu ayrıca test edilir. vLLM sürümünün değişmesi otomatik upgrade veya uyumluluk kabulü doğurmaz.

## Sonraki sıralama

Ortam kanıtı incelendikten sonra tek dar düzeltme veya B kernel testi hazırlanır. Ardından C gerçek iki-GPU iletişim, D OLMoE yükleme/bellek ve E gerçek EP2 kanıtı gelir. İki GPU görünmesi veya TP2 tek başına EP2 değildir: iki scheduler/KV sahibi, yerel tam uzman shard'ları ve gerçek rank'lar arası expert iletişimi gösterilmelidir.

İlk mekanistik workload ancak qualification geçince hazırlanacak: düşük → artan → azalan talep, dengeli/asimetrik rank sahipliği, istek/KV/geçiş kayıtları ve eksik gözlemler. Overload recovery aynı kapasitede toparlanmadır; D2 ekleme değildir. Qualification verisi BLIS kalibrasyonu için kullanılmaz. Stage 1'in gerçek durum/sahiplik anlamını koruma ve koşullu replay/bağımsız tahmin ayrımları geçerlidir.

## Yerel geliştirme kontrolü

```text
python3 -B -m unittest discover -s tests -v
```

Bu testler sahte komut çıktılarıyla paketleme, başarısız önkoşullar, redaction ve UNKNOWN sınırlarını denetler. Gerçek Kaggle GPU sonucu üretmez. `requirements.txt` açıklaması dışında kurulacak paket yoktur.
