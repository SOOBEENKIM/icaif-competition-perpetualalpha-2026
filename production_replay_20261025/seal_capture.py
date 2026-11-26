"""Immutable received-event prefix, independent of later gzip file growth."""
import gzip,hashlib,json,time
from collections import Counter
from .protocol import OUT,save,sha

END=1792915500


def seal():
    while time.time()<END+125:time.sleep(min(30,max(1,END+125-time.time())))
    dest=OUT/'sealed_forward_capture';dest.mkdir(exist_ok=True)
    path=dest/'received_through_fixed_end.jsonl.gz'
    if path.exists():raise FileExistsError('Immutable evaluation prefix already sealed')
    digest=hashlib.sha256();counts=Counter();sources={};last=0
    with gzip.open(path,'wt',compresslevel=3) as writer:
        for f in sorted((OUT/'capture').glob('*.jsonl.gz')):
            sources[str(f.relative_to(OUT))]=sha(f)
            try:
                with gzip.open(f,'rt') as reader:
                    for line in reader:
                        e=json.loads(line)
                        if e['received_ms']>END*1000:continue
                        if e['received_ms']<last:raise ValueError('Nonmonotone capture clock')
                        last=e['received_ms'];counts[e['channel']]+=1
                        encoded=json.dumps(e,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
                        digest.update(encoded.encode());writer.write(encoded)
            except EOFError:pass  # Open final gzip member; complete lines only.
    save(dest/'manifest.json',dict(at=time.time(),end=END,record_counts=dict(counts),last_received_ms=last,
        canonical_event_prefix_sha256=digest.hexdigest(),sealed_file_sha256=sha(path),
        source_files_at_read=sources,note='Original open capture file hashes may change as later events arrive. The canonical fixed-end prefix is preserved here.',orders_sent=0))


if __name__=='__main__':seal()
