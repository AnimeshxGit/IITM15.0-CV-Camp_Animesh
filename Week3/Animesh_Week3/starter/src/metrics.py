"""Classifiers and similarity functions used by evaluate.py."""

import glob
import os
import numpy as np
import torch
from PIL import Image

GALLERY_M = 5
STYLE_TAU = 0.62
# ES and the style centroids use ViT-L/14; SP, BP and the debiaser on-topic
# check use ViT-B/32.
STYLE_CLIP_MODEL = "openai/clip-vit-large-patch14"
STYLE_CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
BASE_CLIP_MODEL = "openai/clip-vit-base-patch32"
BASE_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
DEGENERATE_STD = 1.0


def _as_tensor(out):
    # get_image_features returns a tensor on older transformers, an output
    # object with the projected embedding in pooler_output on newer ones.
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(out, attr, None)
        if val is not None:
            return val
    raise TypeError(f"Unexpected CLIP output type: {type(out)}")


def is_degenerate(pil_image, std_threshold=DEGENERATE_STD):
    """True for blank images, e.g. safety-checker blackouts."""
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    return bool(arr.reshape(-1, 3).std(axis=0).max() < std_threshold)


class CLIPWrapper:
    def __init__(self, clip_model_name=BASE_CLIP_MODEL,
                 revision=BASE_CLIP_REVISION,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.model_name = clip_model_name
        self.model = CLIPModel.from_pretrained(clip_model_name, revision=revision).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, revision=revision)

    @torch.no_grad()
    def embed_image(self, pil_image):
        inputs = self.processor(images=pil_image.convert("RGB"), return_tensors="pt").to(self.device)
        f = _as_tensor(self.model.get_image_features(**inputs))
        return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).cpu()

    @torch.no_grad()
    def embed_text(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True,
                                truncation=True).to(self.device)
        f = _as_tensor(self.model.get_text_features(**inputs))
        return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).cpu()

    @torch.no_grad()
    def clip_score(self, pil_image, text):
        img_emb = self.embed_image(pil_image)
        txt_emb = self.embed_text(text)
        return max(0.0, float(torch.dot(img_emb, txt_emb)))


def image_similarity(clip_wrapper, image_a, image_b):
    emb_a = clip_wrapper.embed_image(image_a)
    emb_b = clip_wrapper.embed_image(image_b)
    return max(0.0, float(torch.dot(emb_a, emb_b)))


class StyleClassifier:
    """Nearest-centroid over the restricted styles, plus an absolute floor.

    Gallery for style_eraser row <case_number> is images/<case_number>_ref_*.png.
    The vote is closed over the five styles, so it cannot tell a suppressed
    style from a mislabelled one; evaluate.py also requires the similarity to
    the TARGET centroid to fall below STYLE_TAU.
    """

    def __init__(self, clip_wrapper, data_dir, style_eraser_rows):
        if STYLE_CLIP_MODEL and STYLE_CLIP_MODEL != getattr(clip_wrapper, "model_name", None):
            self.clip = CLIPWrapper(clip_model_name=STYLE_CLIP_MODEL,
                                    revision=STYLE_CLIP_REVISION)
        else:
            self.clip = clip_wrapper
        self.centroids = {}
        self._build_centroids(data_dir, style_eraser_rows)

    @torch.no_grad()
    def _embed_batch(self, paths, batch_size=16):
        feats = []
        for i in range(0, len(paths), batch_size):
            batch = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = self.clip.processor(images=batch, return_tensors="pt").to(self.clip.device)
            f = _as_tensor(self.clip.model.get_image_features(**inputs))
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(feats, dim=0)

    def _build_centroids(self, data_dir, style_eraser_rows):
        images_dir = os.path.join(data_dir, "images")
        for row in style_eraser_rows:
            artist, case_number = row["target"], row["case_number"]
            paths = sorted(glob.glob(os.path.join(images_dir, f"{case_number}_ref_*.png")))
            if len(paths) != GALLERY_M:
                raise RuntimeError(
                    f"Gallery for {artist} ({case_number}) has {len(paths)} images, "
                    f"expected exactly M = {GALLERY_M}.")
            centroid = self._embed_batch(paths).mean(dim=0)
            self.centroids[artist] = centroid / centroid.norm()

    def classify(self, pil_image):
        """Return (nearest artist, its similarity, {artist: similarity})."""
        emb = self.clip.embed_image(pil_image)
        sims = {artist: float(torch.dot(emb, centroid))
                for artist, centroid in self.centroids.items()}
        best = max(sims, key=sims.get)
        return best, sims[best], sims


class GenderClassifier:
    """FairFace. outputs[7:9] is the gender head, class 0 = Male."""

    def __init__(self, checkpoint_path, shape_predictor_path, face_detector_path,
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        import torchvision
        import dlib
        self.device = device
        self.model = torchvision.models.resnet34(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 18)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model = self.model.to(device).eval()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]),
        ])
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def _aligned_face(self, pil_image):
        import dlib
        arr = np.array(pil_image.convert("RGB"))
        dets = self.cnn_face_detector(arr, 1)
        if len(dets) == 0:
            return None
        shape = self.shape_predictor(arr, dets[0].rect)
        faces = dlib.full_object_detections(); faces.append(shape)
        return Image.fromarray(dlib.get_face_chips(arr, faces, size=300, padding=0.25)[0])

    @torch.no_grad()
    def classify_gender(self, pil_image):
        aligned = self._aligned_face(pil_image)
        if aligned is None:
            return None
        x = self.transform(aligned).unsqueeze(0).to(self.device)
        outputs = self.model(x).cpu().numpy().squeeze()
        return "male" if int(np.argmax(outputs[7:9])) == 0 else "female"
