# OLMoE Kaggle ortam tanılaması

## İlk GPU ortam sonucu ve sıradaki dar düzeltme

21 Kasım 2026 tarihli `env-v0.1.0` kanıtı doğrulandı: arşiv CRC'si, 31/31 iç dosya özeti ve altı kaynak özeti yayımlanan `c1783f1c5fbf310290e17ffc5a7ce729e926fee4` commit'iyle eşleşiyor. İki gerçek T4/SM75 görülüyor; PyTorch CUDA cihaz sorguları çalışıyor. A aşaması yalnız `ModuleNotFoundError: No module named 'vllm'` nedeniyle FAIL. MoE kernel, gerçek NCCL aktarımı ve model-serving çalıştırılmadı.

Güncel Kaggle PyTorch'u `2.11.0+cu130`, Transformers `5.20.2`. [vLLM 0.31.0'ın resmi paket metadatası](https://pypi.org/pypi/vllm/0.31.0/json) PyTorch `2.13.0` ve Transformers `>=5.10.4,<5.18.0` istiyor. Bu fark kurulmuş bir ABI hatası değildir; henüz vLLM kurulmamış.

**Sıradaki notebook: `Kaggle_Isolated_Runtime.ipynb` (`runtime-v0.2.0`).** Yeni private notebook'ta T4 ×2 / Internet On ile Run All kullan. Kaggle ana ortamını değiştirmeden ayrı, yeni bir venv oluşturur; ensurepip kullanmaz. Resmi PyPI'den yalnız hazır ikili paketlerle tek kurulum dener; derleme, alternatif sürüm veya otomatik ikinci deneme yapmaz. Bu birkaç GB indirme/depolama gerektirebilir. Hata olursa ZIP üretip durur. Mevcut ortamları veya önceki kanıtları silmez.

Bu adım yalnız kurulum, `pip check`, `vllm._C` dahil native importlar ve izole A tanılamasıdır. Model indirmez, MoE kernel veya iki-GPU collective çalıştırmaz. Başarılı olsa bile B/C/D/E/M aşamaları çalıştırılmamış kalır. Ana ortamın seçili paket sürümlerini önce/sonra karşılaştırır; tüm dosya sisteminin değişmediğini iddia etmez.

Bana yalnız yeni `moe-runtime-....zip` dosyasını gönder. İçinde tam kurulum stdout/stderr, komutlar, PyPI wheel URL/sürümleri/SHA-256 içeren pip raporu, runner kaynak özetleri, önce/sonra tanılamaları ve sonuç bulunur. vLLM wheel'inin adı ve raporlanan SHA-256'sı incelenen resmi sürümle karşılaştırılır. Bu yayın wheel'inin kabulü, eski kaynak checkout'undaki EngineCore producer patch'inin kurulduğu veya birebir kaynak commit'inin doğrulandığı anlamına gelmez. Transitive paketler pip raporunda kaydedilir; henüz dondurulmuş tam bir lockfile yoktur.

Kaggle'ın sistemde görünen CUDA/NCCL paket metadatası ile PyTorch'un bildirdiği yüklü runtime sürümleri farklı olabilir. İlk kanıt, gerçek yüklenen dosyaların tamamını belirlemediği için bu fark tek başına çakışma olarak yorumlanmıyor. İzole import sonucu görülmeden CUDA veya T4 uyumsuzluğu kabul edilmeyecek.

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
