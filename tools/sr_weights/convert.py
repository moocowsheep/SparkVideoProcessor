#!/usr/bin/env python3
# Copyright 2026 Devin Block
# SPDX-License-Identifier: Apache-2.0

"""Convert pre-trained super-resolution TF frozen graphs into the engine's embedded SRW1 blobs.

Emits engine/operators/resize/sr_weights_data.cpp (committed, so the engine build has no network
or Python dependency). Stdlib only: a minimal protobuf wire-format walker stands in for the
tensorflow/protobuf packages, and the graph is WALKED (Placeholder -> NHWC_output), verifying the
exact op pattern the CUDA executor implements — any structural surprise is a hard error, so a
regenerated blob can't silently disagree with sr_net.cu.

Sources (both Apache-2.0; commits + digests pinned below):
  fsrcnn / fsrcnn-s : github.com/Saafke/FSRCNN_Tensorflow   (x2 models, d=56/s=12/m=4 and d=32/s=5/m=1)
  espcn             : github.com/fannymonori/TF-ESPCN       (x2 model)

Both nets are pixel-shuffle-terminated stacks of SAME/stride-1 convs (the FSRCNN repo replaces the
paper's 9x9 deconv with a 1x1 conv to r^2 channels + depth_to_space; its post-shuffle scalar bias
is folded into that conv's bias here). PReLU appears in the graphs decomposed as
relu(x) + alpha * (x - |x|) * 0.5 and is re-recognized into an activation enum.

Usage:
  python3 tools/sr_weights/convert.py                 # fetch pinned .pb files -> cache/, emit .cpp
  python3 tools/sr_weights/convert.py --cache-only    # use cache/, no network

SRW1 blob layout (little-endian u32 header words, then raw f32 payload per layer):
  magic 'SRW1', version 1, scale, n_layers,
  per layer: kind(0 conv | 1 shuffle2), k, cin, cout, act(0 none|1 prelu|2 relu|3 tanh), has_bias,
  per conv layer payload: weights OIHW [cout][cin][k][k], bias[cout] if has_bias, alpha[cout] if prelu.
"""
import argparse
import hashlib
import os
import struct
import sys
import urllib.request

MODELS = [
    {
        "name": "fsrcnn",
        "repo": "Saafke/FSRCNN_Tensorflow",
        "commit": "6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d",
        "path": "models/FSRCNN_x2.pb",
        "sha256": "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c",
    },
    {
        "name": "fsrcnn-s",
        "repo": "Saafke/FSRCNN_Tensorflow",
        "commit": "6a4812c4ef1c4f5947d79beafa32a05a6eb4a94d",
        "path": "models/FSRCNN-small_x2.pb",
        "sha256": "429e4793d049c1ae16ddbbc322fd11c3c08831c0c20137390b4d098976a2b0d9",
    },
    {
        "name": "espcn",
        "repo": "fannymonori/TF-ESPCN",
        "commit": "5c628eca82028161a53e1265cc3a5b571ab8625f",
        "path": "export/ESPCN_x2.pb",
        "sha256": "59f77351e1d7c0057bf6fe088b4a8a07e42c468c8c8aebb674a6b4ea1823221d",
    },
]
SCALE = 2
ACT = {"none": 0, "prelu": 1, "relu": 2, "tanh": 3}
KIND_CONV, KIND_SHUFFLE = 0, 1


# ---- minimal protobuf wire-format walker (GraphDef -> NodeDef -> AttrValue -> TensorProto) ----

def walk(buf):
    i, n = 0, len(buf)

    def varint():
        nonlocal i
        v = shift = 0
        while True:
            b = buf[i]
            i += 1
            v |= (b & 0x7F) << shift
            if not b & 0x80:
                return v
            shift += 7

    while i < n:
        key = varint()
        field, wire = key >> 3, key & 7
        if wire == 0:
            yield field, wire, varint()
        elif wire == 1:
            yield field, wire, buf[i:i + 8]
            i += 8
        elif wire == 2:
            ln = varint()
            yield field, wire, buf[i:i + ln]
            i += ln
        elif wire == 5:
            yield field, wire, buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")


def parse_shape(buf):
    return [v2 for f, w, v in walk(buf) if f == 2 and w == 2
            for f2, w2, v2 in walk(v) if f2 == 1 and w2 == 0]


def parse_tensor(buf):
    t = {"dtype": None, "shape": [], "floats": []}
    for f, w, v in walk(buf):
        if f == 1 and w == 0:
            t["dtype"] = v
        elif f == 2 and w == 2:
            t["shape"] = parse_shape(v)
        elif f == 4 and w == 2:  # tensor_content: raw LE bytes
            t["floats"] = list(struct.unpack(f"<{len(v) // 4}f", v))
        elif f == 5 and w == 2:  # packed float_val
            t["floats"] += list(struct.unpack(f"<{len(v) // 4}f", v))
        elif f == 5 and w == 5:  # single float_val
            t["floats"].append(struct.unpack("<f", v)[0])
    return t


def parse_node(buf):
    node = {"name": "", "op": "", "inputs": [], "tensor": None}
    for f, w, v in walk(buf):
        if f == 1 and w == 2:
            node["name"] = v.decode()
        elif f == 2 and w == 2:
            node["op"] = v.decode()
        elif f == 3 and w == 2:
            node["inputs"].append(v.decode().split(":")[0].lstrip("^"))
        elif f == 5 and w == 2:  # attr map entry; only "value" (the Const tensor) matters here
            k = av = None
            for f2, w2, v2 in walk(v):
                if f2 == 1 and w2 == 2:
                    k = v2.decode()
                elif f2 == 2 and w2 == 2:
                    av = v2
            if k == "value" and av is not None:
                for f3, w3, v3 in walk(av):
                    if f3 == 8 and w3 == 2:
                        node["tensor"] = parse_tensor(v3)
    return node


def parse_graph(data):
    return [parse_node(v) for f, w, v in walk(data) if f == 1 and w == 2]


# ---- graph walk: extract the conv stack in execution order, verifying the exact pattern ----

def extract_net(nodes, model_name):
    by_name = {n["name"]: n for n in nodes}
    consumers = {}
    for n in nodes:
        for src in n["inputs"]:
            consumers.setdefault(src, []).append(n)

    def const_floats(name, want_rank=None):
        n = by_name[name]
        if n["op"] != "Const" or n["tensor"] is None:
            raise ValueError(f"{model_name}: expected Const '{name}'")
        t = n["tensor"]
        if want_rank is not None and len(t["shape"]) != want_rank:
            raise ValueError(f"{model_name}: '{name}' rank {t['shape']} != {want_rank}")
        return t["floats"], t["shape"]

    placeholders = [n for n in nodes if n["op"] == "Placeholder"]
    if len(placeholders) != 1:
        raise ValueError(f"{model_name}: {len(placeholders)} placeholders")
    cur = placeholders[0]["name"]

    layers = []
    while True:
        nexts = [c for c in consumers.get(cur, []) if c["op"] != "Transpose"]
        if len(nexts) != 1:
            # The only legal fan-out is a decomposed PReLU after a bias add:
            #   add -> { Relu, Abs -> Sub -> Mul(alpha) -> Mul(0.5) } -> Add(join)
            if {c["op"] for c in nexts} != {"Relu", "Abs", "Sub"}:
                raise ValueError(
                    f"{model_name}: node '{cur}' has consumers {[c['op'] for c in nexts]}")
            sub = next(c for c in nexts if c["op"] == "Sub")
            relu = next(c for c in nexts if c["op"] == "Relu")
            mul_a = next(c for c in consumers[sub["name"]] if c["op"] == "Mul")
            alpha_name = next(i for i in mul_a["inputs"] if by_name[i]["op"] == "Const")
            alpha, _ = const_floats(alpha_name)
            if len(alpha) != layers[-1]["cout"]:
                raise ValueError(f"{model_name}: alpha len {len(alpha)} != cout")
            mul_h = next(c for c in consumers[mul_a["name"]] if c["op"] == "Mul")
            half = const_floats(next(i for i in mul_h["inputs"] if by_name[i]["op"] == "Const"))[0]
            if abs(half[0] - 0.5) > 1e-6:
                raise ValueError(f"{model_name}: PReLU scale {half[0]} != 0.5")
            join = next(c for c in consumers[mul_h["name"]] if c["op"] == "Add")
            if relu["name"] not in join["inputs"]:
                raise ValueError(f"{model_name}: PReLU join mismatch")
            layers[-1]["act"] = "prelu"
            layers[-1]["alpha"] = alpha
            cur = join["name"]
            continue
        node = nexts[0]

        if node["op"] == "Conv2D":
            fname = node["inputs"][1]
            wts, shape = const_floats(fname, want_rank=4)
            kh, kw, cin, cout = shape
            if kh != kw:
                raise ValueError(f"{model_name}: non-square kernel {shape}")
            layer = {"kind": KIND_CONV, "k": kh, "cin": cin, "cout": cout, "act": "none",
                     "w_hwio": wts, "bias": None, "alpha": None, "f": fname}
            layers.append(layer)
            cur = node["name"]

        elif node["op"] in ("Add", "BiasAdd") and layers and layers[-1]["bias"] is None \
                and by_name[node["inputs"][1]]["op"] == "Const":
            bias, shape = const_floats(node["inputs"][1])
            if len(bias) != layers[-1]["cout"]:
                raise ValueError(f"{model_name}: bias len {len(bias)} != cout {layers[-1]['cout']}")
            layers[-1]["bias"] = bias
            cur = node["name"]

        elif node["op"] == "Relu":
            layers[-1]["act"] = "relu"
            cur = node["name"]

        elif node["op"] == "DepthToSpace":
            shuffle = {"kind": KIND_SHUFFLE, "k": 0, "cin": SCALE * SCALE, "cout": 1,
                       "act": "none", "w_hwio": None, "bias": None, "alpha": None}
            if layers[-1]["cout"] != SCALE * SCALE:
                raise ValueError(f"{model_name}: shuffle input has {layers[-1]['cout']} channels")
            cur = node["name"]
            tail = [c for c in consumers.get(cur, []) if c["op"] != "Transpose"]
            if len(tail) == 1 and tail[0]["op"] == "BiasAdd":
                # FSRCNN: post-shuffle scalar bias — fold into the pre-shuffle conv's bias.
                bias, _ = const_floats(tail[0]["inputs"][1])
                if len(bias) != 1:
                    raise ValueError(f"{model_name}: post-shuffle bias len {len(bias)}")
                prev = layers[-1]
                base = prev["bias"] or [0.0] * prev["cout"]
                prev["bias"] = [b + bias[0] for b in base]
                cur = tail[0]["name"]
            elif len(tail) == 1 and tail[0]["op"] == "Tanh":
                shuffle["act"] = "tanh"
                cur = tail[0]["name"]
            elif tail:
                raise ValueError(f"{model_name}: unexpected post-shuffle op {tail[0]['op']}")
            layers.append(shuffle)
            if consumers.get(cur, []) and {c["op"] for c in consumers[cur]} != {"Transpose"}:
                raise ValueError(f"{model_name}: output '{cur}' has consumers")
            return layers

        else:
            raise ValueError(f"{model_name}: unexpected op {node['op']} ('{node['name']}')")


def to_oihw(w_hwio, k, cin, cout):
    out = [0.0] * (cout * cin * k * k)
    for kh in range(k):
        for kw in range(k):
            for ci in range(cin):
                base = ((kh * k + kw) * cin + ci) * cout
                for co in range(cout):
                    out[((co * cin + ci) * k + kh) * k + kw] = w_hwio[base + co]
    return out


def serialize(layers):
    blob = struct.pack("<4sIII", b"SRW1", 1, SCALE, len(layers))
    for l in layers:
        has_bias = 1 if l["bias"] is not None else 0
        blob += struct.pack("<IIIIII", l["kind"], l["k"], l["cin"], l["cout"],
                            ACT[l["act"]], has_bias)
        if l["kind"] == KIND_CONV:
            w = to_oihw(l["w_hwio"], l["k"], l["cin"], l["cout"])
            blob += struct.pack(f"<{len(w)}f", *w)
            if has_bias:
                blob += struct.pack(f"<{l['cout']}f", *l["bias"])
            if l["act"] == "prelu":
                blob += struct.pack(f"<{l['cout']}f", *l["alpha"])
    return blob


def describe(layers):
    parts = []
    for l in layers:
        if l["kind"] == KIND_CONV:
            parts.append(f"conv{l['k']}x{l['k']}({l['cin']}->{l['cout']},{l['act']})")
        else:
            parts.append(f"shuffle2({l['act']})")
    return " ".join(parts)


def sanity(layers, name):
    for l in layers:
        if l["kind"] != KIND_CONV:
            continue
        w = l["w_hwio"]
        # Catches mis-parsed bytes (ints read as floats -> denormal/e+30 garbage), not real outliers
        # (espcn has a trained weight at -10.66).
        lo, hi = min(w), max(w)
        if not (-32.0 < lo and hi < 32.0):
            raise ValueError(f"{name}: implausible weight range [{lo}, {hi}]")
    if layers[0]["cin"] != 1:
        raise ValueError(f"{name}: first conv cin {layers[0]['cin']} != 1 (expects luma)")
    if layers[-1]["kind"] != KIND_SHUFFLE:
        raise ValueError(f"{name}: net is not shuffle-terminated")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--out", default=os.path.join(repo, "engine/operators/resize/sr_weights_data.cpp"))
    ap.add_argument("--cache-dir", default=os.path.join(here, "cache"))
    ap.add_argument("--cache-only", action="store_true", help="no network; use cached .pb files")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    entries = []
    for m in MODELS:
        cache = os.path.join(args.cache_dir, os.path.basename(m["path"]))
        if not os.path.exists(cache):
            if args.cache_only:
                raise SystemExit(f"missing {cache} and --cache-only set")
            url = f"https://raw.githubusercontent.com/{m['repo']}/{m['commit']}/{m['path']}"
            print(f"fetch {url}")
            with urllib.request.urlopen(url) as r, open(cache, "wb") as f:
                f.write(r.read())
        data = open(cache, "rb").read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != m["sha256"]:
            raise SystemExit(f"{m['name']}: sha256 mismatch {digest} != {m['sha256']}")
        layers = extract_net(parse_graph(data), m["name"])
        sanity(layers, m["name"])
        blob = serialize(layers)
        print(f"{m['name']:<9} {describe(layers)}  -> {len(blob)} bytes")
        entries.append((m, layers, blob))

    with open(args.out, "w") as f:
        f.write("// Copyright 2026 Devin Block\n")
        f.write("// SPDX-License-Identifier: Apache-2.0\n\n")
        f.write("// GENERATED by tools/sr_weights/convert.py — DO NOT EDIT BY HAND.\n")
        f.write("// Embedded pre-trained x2 super-resolution weights (SRW1 blobs; layout in sr_model.hpp).\n")
        f.write("// Sources (Apache-2.0):\n")
        for m, layers, _ in entries:
            f.write(f"//   {m['name']:<9} github.com/{m['repo']} @{m['commit'][:12]} {m['path']}\n")
            f.write(f"//             sha256 {m['sha256']}\n")
            f.write(f"//             {describe(layers)}\n")
        f.write("#include \"sr_weights_data.hpp\"\n\nnamespace spark::sr {\nnamespace {\n\n")
        for m, _, blob in entries:
            ident = "kBlob_" + m["name"].replace("-", "_")
            f.write(f"const unsigned char {ident}[{len(blob)}] = {{\n")
            for i in range(0, len(blob), 20):
                f.write("    " + ",".join(str(b) for b in blob[i:i + 20]) + ",\n")
            f.write("};\n\n")
        f.write("const EmbeddedModel kModels[] = {\n")
        for m, _, blob in entries:
            ident = "kBlob_" + m["name"].replace("-", "_")
            f.write(f"    {{\"{m['name']}\", {ident}, sizeof({ident})}},\n")
        f.write("};\n\n}  // namespace\n\n")
        f.write("const EmbeddedModel* embedded_models(size_t* count) {\n")
        f.write(f"  *count = {len(entries)};\n  return kModels;\n}}\n\n}}  // namespace spark::sr\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
