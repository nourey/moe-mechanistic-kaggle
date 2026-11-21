# OLMoE Kaggle ortam tanılaması

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
