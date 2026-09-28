import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_ID = "CompVis/stable-diffusion-v1-4"
REVISION = "133a221"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

STYLES = ["Wassily Kandinsky", "Gustav Klimt", "Edvard Munch", "Frida Kahlo", "Amedeo Modigliani"]
ROLES = ["CEO", "Nurse", "Secretary", "Chef", "Construction Worker"]

# The "photograph" anchor forcefully breaks the "painting" prior
ANCHORS = {
    "Wassily Kandinsky": ("photograph", 4.8),
    "Gustav Klimt":      ("lego",       1.4),
    "Edvard Munch":      ("origami",    1.8),
    "Frida Kahlo":       ("blueprint",  1.4),
    "Amedeo Modigliani": ("robot",      1.4),
}

DEBIAS_STRENGTH = {
    "CEO": 3.5, "Nurse": 3.5, "Construction Worker": 6.0, "Secretary": 6.0, "Chef": 6.0,
}

STYLE_VARIANTS = {
    "Wassily Kandinsky": ["Wassily Kandinsky", "wassily kandinsky", "Kandinsky", "kandinsky", "KANDINSKY", "Kandinsky's", "kandinsky's"],
    "Gustav Klimt":      ["Gustav Klimt", "gustav klimt", "Klimt", "klimt", "KLIMT", "Klimt's", "klimt's"],
    "Edvard Munch":      ["Edvard Munch", "edvard munch", "Munch", "munch", "MUNCH", "Munch's", "munch's"],
    "Frida Kahlo":       ["Frida Kahlo", "frida kahlo", "Kahlo", "kahlo", "KAHLO", "Kahlo's", "kahlo's"],
    "Amedeo Modigliani": ["Amedeo Modigliani", "amedeo modigliani", "Modigliani", "modigliani", "MODIGLIANI", "Modigliani's", "modigliani's"],
}

# Strictly restricted to safe roots to prevent Preserver collapse
SUBWORD_HOOKS = {
    "Wassily Kandinsky": ["kand", "insky</w>"],
    "Gustav Klimt":      ["gustav</w>"],
    "Edvard Munch":      ["munch</w>"],
    "Frida Kahlo":       ["frida</w>"],
    "Amedeo Modigliani": [],
}

INTERCEPT_SUBWORDS = True

ROLE_VARIANTS = {
    "CEO":                 ["CEO", "ceo", "Ceo", "C.E.O."],
    "Nurse":               ["Nurse", "nurse", "NURSE"],
    "Secretary":           ["Secretary", "secretary", "SECRETARY"],
    "Chef":                ["Chef", "chef", "CHEF"],
    "Construction Worker": ["Construction Worker", "construction worker", "Construction worker", "CONSTRUCTION WORKER"],
}

def _resolve_anchor(tokenizer, embeddings, phrase):
    ids = tokenizer.encode(phrase, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError("anchor %r tokenised to nothing" % phrase)
    return embeddings[ids].mean(dim=0).clone(), len(ids)

def _atomic_ids(tokenizer, variants, base_vocab_size):
    out = []
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1 and ids[0] >= base_vocab_size:
            out.append(ids[0])
    return sorted(set(out))

def _native_ids(tokenizer, pieces):
    vocab = tokenizer.get_vocab()
    return sorted({vocab[p] for p in pieces if p in vocab})

def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, revision=REVISION, torch_dtype=DTYPE
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    base_vocab_size = len(tokenizer)

    emb = text_encoder.get_input_embeddings().weight.data

    anchor_vecs = {}
    for style, (phrase, scale) in ANCHORS.items():
        vec, n = _resolve_anchor(tokenizer, emb, phrase)
        anchor_vecs[style] = vec * float(scale)

    role_seed = {}
    for role in ROLES:
        ids = tokenizer.encode(role, add_special_tokens=False)
        role_seed[role] = emb[ids].mean(dim=0).clone()

    woman_vec = emb[tokenizer.encode("woman", add_special_tokens=False)[0]].clone()
    man_vec = emb[tokenizer.encode("man", add_special_tokens=False)[0]].clone()

    new_tokens = []
    for v in STYLE_VARIANTS.values(): new_tokens += v
    for v in ROLE_VARIANTS.values(): new_tokens += v
    tokenizer.add_tokens(sorted(set(new_tokens)))
    text_encoder.resize_token_embeddings(len(tokenizer))
    emb = text_encoder.get_input_embeddings().weight.data

    for style in STYLES:
        ids = _atomic_ids(tokenizer, STYLE_VARIANTS[style], base_vocab_size)
        if INTERCEPT_SUBWORDS:
            ids += _native_ids(tokenizer, SUBWORD_HOOKS.get(style, []))
        for tid in sorted(set(ids)):
            emb[tid] = anchor_vecs[style].to(emb.dtype)

    safe_native = {"CEO": ["ceo</w>"], "Nurse": ["nurse</w>"], "Secretary": ["secretary</w>"], "Chef": ["chef</w>"], "Construction Worker": []}

    role_ids, pristine = {}, {}
    for role in ROLES:
        ids = _atomic_ids(tokenizer, ROLE_VARIANTS[role], base_vocab_size)
        for tid in ids:
            emb[tid] = role_seed[role].to(emb.dtype)
        ids = sorted(set(ids + _native_ids(tokenizer, safe_native[role])))
        role_ids[role] = ids
        pristine[role] = {tid: emb[tid].clone() for tid in ids}

    pipe.las_role_ids = role_ids
    pipe.las_pristine = pristine
    pipe.las_gender = {"woman": woman_vec.to(emb.dtype), "man": man_vec.to(emb.dtype)}
    return pipe

@torch.no_grad()
def generate(pipe, prompt: str, seed: int) -> Image.Image:
    emb = pipe.text_encoder.get_input_embeddings().weight.data

    even = (seed % 2 == 0)
    target = pipe.las_gender["woman"] if even else pipe.las_gender["man"]
    opposite = pipe.las_gender["man"] if even else pipe.las_gender["woman"]
    direction = target - opposite

    try:
        for role, ids in pipe.las_role_ids.items():
            k = DEBIAS_STRENGTH[role]
            for tid in ids:
                emb[tid] = pipe.las_pristine[role][tid] + k * direction

        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        output = pipe(prompt, num_inference_steps=50, generator=generator)
        image = output.images[0]
        
        flagged = getattr(output, "nsfw_content_detected", None)
        if flagged and flagged[0]:
            print(f"[NSFW FILTER] blocked prompt={prompt!r} seed={seed}")
    finally:
        for role, ids in pipe.las_role_ids.items():
            for tid in ids:
                emb[tid] = pipe.las_pristine[role][tid]

    return image