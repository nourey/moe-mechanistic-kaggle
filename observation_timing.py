"""CUDA-stream interval receipts; no profiler, cross-device clock or network claim."""
import math


def start_interval(torch):
    from vllm.utils.torch_utils import current_stream
    stream = torch.cuda.current_stream()
    native = current_stream()
    event = torch.cuda.Event(enable_timing=True)
    event.record(stream)
    return dict(_start=event, stream=stream.cuda_stream, device=torch.cuda.current_device(),
                native_stream_matches=native.cuda_stream == stream.cuda_stream)


def end_interval(torch, row):
    stream = torch.cuda.current_stream()
    row['end_stream_matches'] = stream.cuda_stream == row['stream']
    row['end_device_matches'] = torch.cuda.current_device() == row['device']
    row['_end'] = torch.cuda.Event(enable_timing=True)
    row['_end'].record(stream)


def finish_interval(row):
    start = row.pop('_start'); end = row.pop('_end')
    ready = start.query() and end.query()
    compatible=ready and row['native_stream_matches'] and row['end_stream_matches'] and row.get('end_device_matches',False)
    try:value = start.elapsed_time(end) if compatible else None
    except Exception as error:
        row['error']=repr(error);value=None
    valid = compatible and value is not None and math.isfinite(value) and value >= 0
    row.update(status='OBSERVED_STREAM_INTERVAL' if valid else 'UNSUPPORTED', milliseconds=value if valid else None,
        meaning='One device stream interval, including queue/peer waits and possible host enqueue gaps; not pure network or kernel time')
    return row


def summarize(values):
    """Keep small-sample variability visible; no fitted significance/equivalence threshold."""
    import statistics
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values: return dict(n=0, median=None, minimum=None, maximum=None, MAD=None)
    median = statistics.median(values)
    return dict(n=len(values), median=median, minimum=min(values), maximum=max(values),
                MAD=statistics.median(abs(v-median) for v in values))
