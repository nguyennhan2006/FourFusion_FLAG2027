# ============================================================================ VoxBlink media helpers (FLAG_46): shard index, timestamps, cutting
# Source: the unofficial Hugging Face crawl adollahamini1998/VoxBlink-Video-clean-8-{0..7} (whole YouTube shorts, h264 <= 720p,
# 25 fps, AAC 16 kHz) + the official VoxBlink annotations (timestamp.tar.gz: one file per utterance, per-frame face box;
# meta/spk2lan, spk2loc, spk2gender). VoxBlink: CC BY-NC-SA 4.0, the authors forbid redistribution: research use inside the
# challenge, outputs kept private, source declared in the system description.
# Every function here works on one video at a time (no statistic of other samples).
import tarfile as _tarfile


def tar_parse_header(h):
    """512-byte ustar header -> (name, size, typeflag). Sizes in base-256 (GNU) are handled."""
    name = h[0:100].rstrip(b"\0").decode("utf8", "replace")
    prefix = h[345:500].rstrip(b"\0").decode("utf8", "replace")
    sz = h[124:136]
    size = int.from_bytes(sz[1:], "big") if sz[0] & 0x80 else int(sz.rstrip(b"\0 ").decode() or "0", 8)
    return (prefix + "/" + name if prefix else name), size, (chr(h[156]) if h[156] else "0")


def tar_list_remote(session, url, max_members=None):
    """Member list of a remote plain tar by HTTP range reads of the headers only: [(name, data offset, size)].
    PAX per-file headers ('x') are followed to the real header; their 'path' record overrides the name."""
    off, out = 0, []
    while max_members is None or len(out) < max_members:
        b = session.get(url, headers={"Range": f"bytes={off}-{off + 2047}"}, timeout=120).content
        if len(b) < 512 or b[:512] == b"\0" * 512:
            break
        name, size, typ = tar_parse_header(b[:512])
        if typ in ("x", "g"):
            pax = b[512:512 + size] if 512 + size <= len(b) else session.get(
                url, headers={"Range": f"bytes={off + 512}-{off + 512 + size - 1}"}, timeout=120).content
            path = None
            for rec in pax.decode("utf8", "replace").split("\n"):
                if " path=" in rec:
                    path = rec.split(" path=", 1)[1]
            off2 = off + 512 + ((size + 511) // 512) * 512
            h2 = b[off2 - off:off2 - off + 512] if off2 - off + 512 <= len(b) else session.get(
                url, headers={"Range": f"bytes={off2}-{off2 + 511}"}, timeout=120).content
            name2, size2, _ = tar_parse_header(h2)
            if typ == "x":
                out.append((path or name2, off2 + 512, size2))
            off = off2 + 512 + ((size2 + 511) // 512) * 512
        else:
            if typ in ("0", "\0"):
                out.append((name, off + 512, size))
            off = off + 512 + ((size + 511) // 512) * 512
    return out


def parse_timestamp(text):
    """Official VoxBlink timestamp file -> (frames int array, boxes (n, 4) x, y, w, h). Header: 'Identity', 'Reference',
    blank, 'FRAME X Y W H', blank, then one tab-separated line per frame (25 fps)."""
    fr, bx = [], []
    for line in text.splitlines():
        p = line.strip().split("\t")
        if len(p) == 5 and p[0].isdigit():
            fr.append(int(p[0])); bx.append([int(float(v)) for v in p[1:]])
    return np.asarray(fr, int), np.asarray(bx, int).reshape(-1, 4)


def loose_crop(rgb, box, scale, out=224):
    """Square crop of side scale * max(w, h) centred on the face box, padded with black outside the image, resized to out:
    the loose head-and-shoulders framing of MAV-Celeb / VoxCeleb face tracks (the official VoxBlink crop is tight, 112)."""
    from PIL import Image
    x, y, w, h = [float(v) for v in box]
    cx, cy, side = x + w / 2, y + h / 2, scale * max(w, h)
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2)); s = int(round(side))
    H, W = rgb.shape[:2]
    canvas = np.zeros((s, s, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + s), min(H, y0 + s)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = rgb[sy0:sy1, sx0:sx1]
    inside = (sx1 - sx0) * (sy1 - sy0) / float(s * s)
    return Image.fromarray(canvas).resize((out, out), Image.BICUBIC), inside


def decode_video(mp4, frames_wanted, sr=16000):
    """One pass over a short: the RGB frames at the wanted indices (decoded in order, no seeking: shorts are <= 60 s) and the
    whole audio track as mono float32 at sr. -> ({frame index: HxWx3 uint8}, audio)."""
    import av
    want, got = sorted(set(int(f) for f in frames_wanted)), {}
    audio = []
    with av.open(io.BytesIO(mp4)) as c:
        vs = c.streams.video[0] if c.streams.video else None
        if vs is not None:
            vs.thread_type = "AUTO"
            last = max(want) if want else -1
            for k, fr in enumerate(c.decode(vs)):
                if k in want:
                    got[k] = fr.to_ndarray(format="rgb24")
                if k >= last:
                    break
    with av.open(io.BytesIO(mp4)) as c:
        if c.streams.audio:
            res = av.AudioResampler(format="flt", layout="mono", rate=sr)
            for fr in c.decode(c.streams.audio[0]):
                audio += [o.to_ndarray().reshape(-1) for o in res.resample(fr)]
            audio += [o.to_ndarray().reshape(-1) for o in res.resample(None)]
    return got, (np.concatenate(audio).astype(np.float32) if audio else np.zeros(0, np.float32))
