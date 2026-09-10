"""Encodeur texte de remplacement (Models > Checkpoints > Text encoder), Z-Image.

Z-Image lit l'avant-dernier etat cache (hidden_states[-2]) de son encodeur Qwen3-4B, et
le transformer l'attend large de 2560 (cap_feat_dim): un autre encodeur ne se branche
que s'il a la meme famille, la meme largeur et le meme nombre de couches. Le refus doit
le dire AVANT de lire 8 Go.

Ces tests verrouillent aussi ce qui rendrait l'option dangereuse en silence:
  - la classe vient du model_index.json du repo (Qwen3Model), pas de
    text_encoder/config.json (Qwen3ForCausalLM), et un checkpoint Qwen3ForCausalLM s'y
    charge;
  - un changement d'encodeur libere le pipeline, et free_vram oublie l'encodeur actif;
  - _ensure_base passe l'encodeur a from_pretrained, et retombe sur celui du repo, sans
    planter, s'il ne convient pas ou s'il echoue au chargement;
  - img2img / inpaint (from_pipe) reprennent l'objet encodeur du base;
  - les metadonnees nomment l'encodeur qui a REELLEMENT tourne, par son nom de dossier
    et jamais par son chemin (qui finirait dans les PNG partages); Omni n'en dit rien;
  - la file garde l'encodeur du job; l'UI ne memorise qu'un encodeur valide.

CPU seulement, sans reseau, sans vrai modele: la config de l'encodeur du repo de base
est remplacee, les modeles sont minuscules et construits a la volee.
Run:  .venv/Scripts/python tests/test_text_encoder.py
"""
import json
import os
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # jamais de GPU ici
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch

import cz_imageio
import cz_pipeline as P

# Encodeur de Z-Image-Turbo / Z-Image (text_encoder/config.json du repo).
ZIMAGE_TE = {"model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36,
             "architectures": ["Qwen3ForCausalLM"]}
QWEN8B = {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 36,
          "architectures": ["Qwen3ForCausalLM"]}


def _folder(cfg, sub=None, name="enc"):
    root = tempfile.mkdtemp(prefix="te_")
    d = os.path.join(root, name)
    p = os.path.join(d, sub) if sub else d
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return d


def _zimage_like_base():
    """Dossier au format du repo Z-Image: model_index.json dit Qwen3Model,
    text_encoder/config.json dit Qwen3ForCausalLM."""
    base = tempfile.mkdtemp(prefix="base_")
    with open(os.path.join(base, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump({"_class_name": "ZImagePipeline",
                   "text_encoder": ["transformers", "Qwen3Model"]}, f)
    os.makedirs(os.path.join(base, "text_encoder"))
    with open(os.path.join(base, "text_encoder", "config.json"), "w", encoding="utf-8") as f:
        json.dump(ZIMAGE_TE, f)
    return base


def _tiny_qwen3_cfg():
    from transformers import Qwen3Config
    return Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                       head_dim=8, max_position_embeddings=64)


class _Base:
    """Remplace la config de l'encodeur du repo de base (pas de reseau, pas de HF)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __enter__(self):
        self.old = P._base_text_encoder_config
        P._base_text_encoder_config = lambda base=None: self.cfg

    def __exit__(self, *a):
        P._base_text_encoder_config = self.old


class _Saved:
    """Sauve puis restaure des globaux de cz_pipeline."""

    def __init__(self, *names):
        self.names = names

    def __enter__(self):
        self.vals = {n: getattr(P, n) for n in self.names}

    def __exit__(self, *a):
        for n, v in self.vals.items():
            setattr(P, n, v)


def test_same_architecture_is_accepted():
    with _Base(ZIMAGE_TE):
        assert P._text_encoder_problem(_folder(ZIMAGE_TE)) is None
        # poids dans un sous-dossier text_encoder/ (copie d'un repo diffusers)
        assert P._text_encoder_problem(_folder(ZIMAGE_TE, "text_encoder")) is None
        # sauve en Qwen3Model plutot qu'en Qwen3ForCausalLM: meme encodeur
        assert P._text_encoder_problem(
            _folder({**ZIMAGE_TE, "architectures": ["Qwen3Model"]})) is None
    print("OK test_same_architecture_is_accepted")


def test_a_wider_encoder_is_refused_with_both_widths():
    with _Base(ZIMAGE_TE):
        why = P._text_encoder_problem(_folder(QWEN8B))
    assert why and "4096" in why and "2560" in why, why
    print("OK test_a_wider_encoder_is_refused_with_both_widths")


def test_other_family_and_layer_count_are_refused():
    with _Base(ZIMAGE_TE):
        why = P._text_encoder_problem(_folder({"model_type": "t5", "d_model": 2560,
                                               "num_layers": 36}))
        assert why and "t5" in why and "qwen3" in why, why
        why = P._text_encoder_problem(_folder({**ZIMAGE_TE, "num_hidden_layers": 28}))
        assert why and "28" in why and "36" in why, why
    print("OK test_other_family_and_layer_count_are_refused")


def test_gguf_single_file_and_empty_folder_are_refused_with_the_reason():
    with _Base(ZIMAGE_TE):
        assert "GGUF" in P._text_encoder_problem(r"F:\x\qwen3-4b-q8_0.gguf")
        assert "FOLDER" in P._text_encoder_problem(r"F:\x\qwen3_4b.safetensors")
        assert "config.json" in P._text_encoder_problem(tempfile.mkdtemp())
        # un fichier present, quelle que soit son extension
        f = os.path.join(tempfile.mkdtemp(), "weights.dat")
        open(f, "wb").close()
        assert "FOLDER" in P._text_encoder_problem(f)
        # chemin absent de cette machine: refuse sans passer par le reseau
        why = P._text_encoder_problem(r"Z:\nowhere\qwen3-abl")
        assert why and "neither a folder" in why, why
    print("OK test_gguf_single_file_and_empty_folder_are_refused_with_the_reason")


def test_hf_ids_may_carry_a_subfolder():
    assert P._split_hf_src("owner/repo") == ("owner/repo", None)
    assert P._split_hf_src("owner/repo/text_encoder") == ("owner/repo", "text_encoder")
    assert P._split_hf_src("owner/repo/a/b") == ("owner/repo", "a/b")
    print("OK test_hf_ids_may_carry_a_subfolder")


def test_the_class_comes_from_the_base_repo_model_index():
    """model_index.json dit Qwen3Model, text_encoder/config.json dit Qwen3ForCausalLM:
    diffusers charge la premiere, c'est elle que le pipeline attend."""
    base = _zimage_like_base()
    cls = P._encoder_class(base)
    assert cls.__name__ == "Qwen3Model", cls
    # la config de reference se lit dans le dossier local du repo de base
    assert P._base_text_encoder_config(base) == ZIMAGE_TE
    assert P._text_encoder_problem(_folder(ZIMAGE_TE), base) is None
    why = P._text_encoder_problem(_folder(QWEN8B), base)
    assert why and "4096" in why and "2560" in why, why
    print("OK test_the_class_comes_from_the_base_repo_model_index")


def test_a_causal_lm_checkpoint_loads_into_the_pipeline_class():
    """Un Qwen3 'abliterated' est publie en Qwen3ForCausalLM: il doit se charger en
    Qwen3Model (classe du pipeline), poids du tronc intacts, en DTYPE."""
    from transformers import Qwen3ForCausalLM, Qwen3Model
    torch.manual_seed(0)
    src = Qwen3ForCausalLM(_tiny_qwen3_cfg())
    enc_dir = os.path.join(tempfile.mkdtemp(prefix="te_"), "qwen3-tiny-abl")
    src.save_pretrained(os.path.join(enc_dir, "text_encoder"))
    m = P._load_text_encoder(enc_dir, _zimage_like_base())
    assert type(m) is Qwen3Model, type(m)
    assert m.embed_tokens.weight.dtype == P.DTYPE, m.embed_tokens.weight.dtype
    assert torch.equal(m.embed_tokens.weight, src.model.embed_tokens.weight.to(P.DTYPE))
    assert torch.equal(m.layers[1].mlp.up_proj.weight,
                       src.model.layers[1].mlp.up_proj.weight.to(P.DTYPE))
    print("OK test_a_causal_lm_checkpoint_loads_into_the_pipeline_class")


def test_changing_the_encoder_frees_the_pipe():
    with _Saved("TEXT_ENCODER", "_BASE_PIPE", "_DERIVED", "_LOADED_KEY",
                "_TEXT_ENCODER_ACTIVE"):
        P.TEXT_ENCODER = ""
        P._BASE_PIPE = object()
        P._DERIVED = {"txt2img": P._BASE_PIPE, "img2img": object()}
        P._LOADED_KEY = ("repo", None, "none")
        P._TEXT_ENCODER_ACTIVE = r"D:\enc\old"
        P.set_text_encoder(r"D:\enc\qwen3-abl")
        assert P.TEXT_ENCODER == r"D:\enc\qwen3-abl"
        assert P._BASE_PIPE is None and P._DERIVED == {} and P._LOADED_KEY is None, \
            "le pipeline (et ses derives) doit etre libere"
        assert P._TEXT_ENCODER_ACTIVE == "", "free_vram doit oublier l'encodeur charge"
        # meme valeur: rien ne bouge, pas de rechargement inutile
        sentinel = P._BASE_PIPE = object()
        P.set_text_encoder(r"  D:\enc\qwen3-abl ")
        assert P._BASE_PIPE is sentinel
    print("OK test_changing_the_encoder_frees_the_pipe")


class _FakeZPipe:
    """ZImagePipeline factice: note les kwargs de from_pretrained, ne charge rien."""
    calls = []

    def __init__(self, kw):
        self.kw = kw
        self.scheduler = None
        self.vae = None
        self.text_encoder = kw.get("text_encoder", "BASE-ENCODER")

    @classmethod
    def from_pretrained(cls, repo, **kw):
        cls.calls.append((repo, kw))
        return cls(kw)

    def to(self, *a, **k):
        return self


def test_ensure_base_passes_the_encoder_or_falls_back():
    import diffusers
    real = diffusers.ZImagePipeline
    enc = object()

    def _raise(exc):
        def f(*a, **k):
            raise exc
        return f

    def run(check=lambda src, base=None: None, load=lambda src, base=None: enc):
        P.free_vram()
        _FakeZPipe.calls.clear()
        P._text_encoder_problem = check
        P._load_text_encoder = load
        P._ensure_base()                      # ne doit jamais lever
        (repo, kw), = _FakeZPipe.calls
        assert repo == P.BASE_REPO and kw.get("torch_dtype") == P.DTYPE, (repo, kw)
        return kw

    with _Saved("_BASE_PIPE", "_DERIVED", "_LOADED_KEY", "_BASE_SCHED_CONFIG",
                "_APPLIED_LORAS", "TEXT_ENCODER", "_TEXT_ENCODER_ACTIVE",
                "ZIMAGE_TRANSFORMER", "LORAS", "PROMPT_LORAS", "LOAD_PROGRESS_ENABLED",
                "_text_encoder_problem", "_load_text_encoder"):
        try:
            diffusers.ZImagePipeline = _FakeZPipe
            P.ZIMAGE_TRANSFORMER = None
            P.LORAS, P.PROMPT_LORAS = [], []
            P.LOAD_PROGRESS_ENABLED = False
            # 1. aucun encodeur choisi: from_pretrained exactement comme avant
            P.TEXT_ENCODER = ""
            kw = run(check=_raise(AssertionError("pas de verification sans encodeur")))
            assert "text_encoder" not in kw and P._TEXT_ENCODER_ACTIVE == "", kw
            # 2. encodeur valide: passe a from_pretrained, marque actif
            P.TEXT_ENCODER = r"C:\Users\someone\text_encoders\qwen3-abl"
            kw = run()
            assert kw.get("text_encoder") is enc, kw
            assert P._TEXT_ENCODER_ACTIVE == P.TEXT_ENCODER
            assert P._BASE_PIPE.text_encoder is enc
            # 3. ne convient pas (repo change depuis le choix): ecarte, l'encodeur du
            #    repo tourne, les metadonnees le disent
            kw = run(check=lambda src, base=None: "hidden size 4096, and x's encoder is 2560 wide")
            assert "text_encoder" not in kw and P._TEXT_ENCODER_ACTIVE == "", kw
            m = P._gen_meta("txt2img", "p")
            assert m.get("text_encoder_not_applied") == "qwen3-abl" and "text_encoder" not in m, m
            # 4. echec au chargement (disque debranche...): idem, pas d'exception
            kw = run(load=_raise(OSError("disk gone")))
            assert "text_encoder" not in kw and P._TEXT_ENCODER_ACTIVE == "", kw
            # 5. la verification elle-meme leve (config corrompue): idem
            kw = run(check=_raise(ValueError("bad config")))
            assert "text_encoder" not in kw and P._TEXT_ENCODER_ACTIVE == "", kw
        finally:
            diffusers.ZImagePipeline = real
            P.free_vram()
    print("OK test_ensure_base_passes_the_encoder_or_falls_back")


def test_derived_pipes_share_the_base_encoder():
    """img2img / inpaint derivent du base via from_pipe (le vrai de diffusers): ils
    doivent reprendre l'objet encodeur du base -- celui de remplacement quand il est
    charge -- et non en charger un autre."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from transformers import Qwen3Model
    torch.manual_seed(0)
    enc = Qwen3Model(_tiny_qwen3_cfg())
    base = ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=AutoencoderKL(),
                          text_encoder=enc, tokenizer=None, transformer=None)
    with _Saved("_BASE_PIPE", "_DERIVED", "_LOADED_KEY", "_BASE_SCHED_CONFIG",
                "_APPLIED_LORAS", "ZIMAGE_TRANSFORMER", "LORAS", "PROMPT_LORAS"):
        try:
            P.ZIMAGE_TRANSFORMER = None
            P.LORAS, P.PROMPT_LORAS, P._APPLIED_LORAS = [], [], []
            P._BASE_SCHED_CONFIG = None
            P._BASE_PIPE = base
            P._DERIVED = {"txt2img": base}
            P._LOADED_KEY = (P.BASE_REPO, P.ZIMAGE_TRANSFORMER, P.OFFLOAD_MODE)
            for kind, cls in (("img2img", "ZImageImg2ImgPipeline"),
                              ("inpaint", "ZImageInpaintPipeline")):
                d = P.get_pipe(kind)
                assert type(d).__name__ == cls, type(d)
                assert d.text_encoder is enc, f"{kind}: encodeur non partage avec le base"
            assert P._BASE_PIPE is base, "deriver ne doit pas recharger le base"
        finally:
            P.free_vram()
    print("OK test_derived_pipes_share_the_base_encoder")


def test_metadata_names_the_encoder_that_ran_and_never_its_path():
    path = r"C:\Users\someone\models\text_encoders\qwen3-4b-abliterated"
    with _Saved("TEXT_ENCODER", "_TEXT_ENCODER_ACTIVE"):
        P.TEXT_ENCODER = P._TEXT_ENCODER_ACTIVE = path
        m = P._gen_meta("txt2img", "p")
        assert m["text_encoder"] == "qwen3-4b-abliterated", m
        assert "someone" not in json.dumps(m), "chemin local dans les metadonnees"
        # Omni a son propre encodeur: rien a declarer
        m = P._gen_meta("omni", "p")
        assert "text_encoder" not in m and "text_encoder_not_applied" not in m, m
        # demande mais ecarte au chargement: nomme a part
        P._TEXT_ENCODER_ACTIVE = ""
        m = P._gen_meta("img2img", "p")
        assert "text_encoder" not in m and m["text_encoder_not_applied"] == "qwen3-4b-abliterated", m
        P.TEXT_ENCODER = ""
        m = P._gen_meta("txt2img", "p")
        assert "text_encoder" not in m and "text_encoder_not_applied" not in m, m
    assert P._encoder_label(r"D:\m\ponpoke-uncensored\text_encoder") == "ponpoke-uncensored"
    assert P._encoder_label("owner/repo/sub") == "owner/repo/sub"
    line = cz_imageio._a1111_parameters({"prompt": "p", "model": "x", "text_encoder": "qwen3-4b-abliterated"})
    assert "Text encoder: qwen3-4b-abliterated" in line, line
    assert "Text encoder" not in cz_imageio._a1111_parameters({"prompt": "p", "model": "x"})
    print("OK test_metadata_names_the_encoder_that_ran_and_never_its_path")


def test_the_list_finds_encoder_folders():
    d = _folder(ZIMAGE_TE, name="qwen3-4b-abliterated")
    root = os.path.dirname(d)
    os.makedirs(os.path.join(root, "empty"))
    # a cote du dossier de checkpoints EXTRA (bibliotheque partagee, autre disque)
    lib = tempfile.mkdtemp(prefix="lib_")
    os.makedirs(os.path.join(lib, "checkpoints"))
    e = os.path.join(lib, "text_encoders", "qwen3-4b-ft")
    os.makedirs(os.path.join(e, "text_encoder"))
    with open(os.path.join(e, "text_encoder", "config.json"), "w", encoding="utf-8") as f:
        json.dump(ZIMAGE_TE, f)
    with _Saved("TEXT_ENCODERS_DIR", "CHECKPOINTS_EXTRA_DIR"):
        P.TEXT_ENCODERS_DIR = root
        P.CHECKPOINTS_EXTRA_DIR = os.path.join(lib, "checkpoints")
        found = P.list_text_encoders()
    assert d in found, found
    assert e in found, found
    assert not any(f.endswith("empty") for f in found), found
    assert len(found) == len(set(found)), found
    print("OK test_the_list_finds_encoder_folders")


def test_the_queue_keeps_the_encoder():
    import cz_ui as U
    calls = []
    old = (P.TEXT_ENCODER, P.set_text_encoder)
    try:
        P.TEXT_ENCODER = r"D:\enc\qwen3-abl"
        ms = U._q_model_state()
        assert ms["text_encoder"] == r"D:\enc\qwen3-abl", ms
        P.set_text_encoder = lambda s: calls.append(s)
        U._q_restore_model_state(ms)
        assert calls == [r"D:\enc\qwen3-abl"], calls
        # snapshot d'avant l'option: on ne touche pas a l'encodeur courant
        calls.clear()
        U._q_restore_model_state({k: v for k, v in ms.items() if k != "text_encoder"})
        assert calls == [], calls
    finally:
        P.TEXT_ENCODER, P.set_text_encoder = old
    print("OK test_the_queue_keeps_the_encoder")


def test_the_ui_saves_only_a_valid_encoder():
    import cz_ui as U
    saved, calls = [], []
    old = (U._save_prefs_keys, P.set_text_encoder, P.TEXT_ENCODER, P.list_text_encoders)
    try:
        U._save_prefs_keys = lambda d: saved.append(d)
        P.set_text_encoder = lambda s: calls.append(s)
        P.TEXT_ENCODER = ""
        with _Base(ZIMAGE_TE):
            msg = U._ui_set_text_encoder(_folder(QWEN8B))
            assert "not applied" in msg and "4096" in msg, msg
            assert saved == [] and calls == [], (saved, calls)
            good = _folder(ZIMAGE_TE, name="qwen3-4b-abliterated")
            msg = U._ui_set_text_encoder(good)
            assert "qwen3-4b-abliterated" in msg and "reloads" in msg, msg
            assert calls == [good] and saved == [{"text_encoder": good}], (saved, calls)
            U._ui_set_text_encoder("")
            assert calls[-1] == "" and saved[-1] == {"text_encoder": ""}, (saved, calls)
        # dropdown: le defaut, les dossiers trouves, et la valeur collee d'ailleurs
        P.list_text_encoders = lambda: [good]
        P.TEXT_ENCODER = "owner/repo"
        ch = U._te_choices()
        assert ch[0][1] == "" and (("qwen3-4b-abliterated", good) in ch), ch
        assert ("owner/repo", "owner/repo") in ch, ch
    finally:
        U._save_prefs_keys, P.set_text_encoder, P.TEXT_ENCODER, P.list_text_encoders = old
    print("OK test_the_ui_saves_only_a_valid_encoder")


def test_config_sample_documents_the_keys():
    with open(os.path.join(HERE, "config-sample.txt"), encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ("text_encoder", "text_encoders_dir"):
        assert k in cfg and f"_{k}_help" in cfg, k
        assert cfg[k] == "", f"{k}: vide par defaut (l'encodeur du repo de base)"
    print("OK test_config_sample_documents_the_keys")


def test_default_picked_in_the_ui_survives_a_restart():
    """Choisir "Default" ecrit "" dans les preferences: au redemarrage, une valeur de
    config.txt ne doit pas revenir par-dessus. L'environnement gagne toujours."""
    cfg = {"text_encoder": r"D:\enc\from-config"}
    assert P._resolve_text_encoder({}, {}, cfg) == r"D:\enc\from-config"
    assert P._resolve_text_encoder({}, {"text_encoder": ""}, cfg) == ""
    assert P._resolve_text_encoder({}, {"text_encoder": r"D:\enc\ui"}, cfg) == r"D:\enc\ui"
    assert P._resolve_text_encoder({"ZIMAGE_TEXT_ENCODER": r"D:\enc\env"},
                                   {"text_encoder": ""}, cfg) == r"D:\enc\env"
    print("OK test_default_picked_in_the_ui_survives_a_restart")


def test_compatible_encoders_in_the_hf_cache_are_listed():
    """Un encodeur telecharge depuis HF vit dans le cache HF: la liste doit le montrer.
    Pas un pipeline diffusers, pas une config sans poids; une autre taille est nommee a cote."""
    import json as _json
    import os as _os
    import tempfile as _tempfile
    ref = {"model_type": "fam", "hidden_size": 64, "num_hidden_layers": 2}
    wide = {"model_type": "fam", "hidden_size": 128, "num_hidden_layers": 2}
    root = _tempfile.mkdtemp(prefix="hfcache_")

    def snap(repo, sub=None, cfg=ref, weights=True, pipeline=False):
        d = _os.path.join(root, "models--" + repo.replace("/", "--"), "snapshots", "r1")
        p = _os.path.join(d, sub) if sub else d
        _os.makedirs(p, exist_ok=True)
        with open(_os.path.join(p, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg, f)
        if weights:
            open(_os.path.join(p, "model.safetensors"), "wb").close()
        if pipeline:
            with open(_os.path.join(d, "model_index.json"), "w", encoding="utf-8") as f:
                f.write("{}")

    snap("a/fits")
    snap("b/fits-in-sub", sub="enc")
    snap("c/wider", cfg=wide)
    snap("d/pipeline", sub="text_encoder", pipeline=True)
    snap("e/config-only", weights=False)
    snap("f/no-shape", cfg={"_class_name": "AutoencoderKL"})
    old = (P._hf_cache_dir, P._base_text_encoder_config)
    try:
        P._hf_cache_dir = lambda: root
        P._base_text_encoder_config = lambda base=None: ref
        got = [v for _l, v in P.list_cached_text_encoders()]
        other, width = P.cached_text_encoder_mismatches()
        import cz_ui as U
        hint = U._te_hint()
        choices = [v for _l, v in U._te_choices()]
    finally:
        P._hf_cache_dir, P._base_text_encoder_config = old
    assert got == ["a/fits", "b/fits-in-sub/enc"], got
    assert all(v in choices for v in got), choices
    assert [h for h, _w in other] == ["c/wider"] and width == 64, (other, width)
    assert "128" in hint and "64" in hint and "c/wider" in hint, hint
    print("OK test_compatible_encoders_in_the_hf_cache_are_listed")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("All text-encoder tests passed.")
