"""Demo: classifica um frame e compara com a etiqueta de shot size conhecida.

Usa o ensemble das seeds (média das probabilidades, sem TTA). Mostra a classe
prevista, as duas mais prováveis, quantos modelos concordam e avisa quando a
decisão é uma fronteira ambígua entre classes vizinhas.

Uso:
  # modo interativo (os modelos carregam uma vez; depois é só ir escrevendo)
  python scripts/demo_predict.py
  > /caminho/frame.jpg | close up        (etiqueta conhecida, opcional)
  > /caminho/outro_frame.jpg             (sem etiqueta)

  # uma imagem só
  python scripts/demo_predict.py /caminho/frame.jpg --label "close up"

Se a imagem estiver nos CSVs de treino/val/teste, a etiqueta é encontrada pelo
nome do ficheiro e o script avisa se o modelo a viu no treino.
Aceita duas etiquetas separadas por vírgula (ex.: "close up, medium close up").
"""

from __future__ import annotations

import argparse
import glob
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUNS = "data/processed/finetune_shotsize_protocol/seed*"
DEFAULT_ORDER = [
    "extreme close up",
    "close up",
    "medium close up",
    "medium",
    "medium wide",
    "wide",
    "extreme wide",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("images", nargs="*", help="imagens a classificar (vazio = modo interativo)")
    p.add_argument("--label", default=None, help="etiqueta conhecida (só com uma imagem)")
    p.add_argument("--runs", nargs="+", default=[DEFAULT_RUNS], help="pastas das seeds (ou globs)")
    p.add_argument("--single", action="store_true", help="usar só a primeira seed (carrega mais depressa)")
    p.add_argument("--class-order", nargs="+", default=DEFAULT_ORDER)
    p.add_argument(
        "--ambiguity-margin",
        type=float,
        default=0.25,
        help="avisa se as duas classes mais prováveis são vizinhas e a diferença é menor que isto",
    )
    p.add_argument("--show-models", action="store_true", help="mostra a previsão de cada seed")
    return p.parse_args()


def normalize(name: str) -> str:
    return " ".join(name.lower().replace("-", " ").replace("_", " ").split())


def parse_known(text: str, classes: list[str]) -> list[str]:
    lookup = {normalize(c): c for c in classes}
    parts = [normalize(part) for part in text.split(",") if part.strip()]
    bad = [part for part in parts if part not in lookup]
    if not parts or bad:
        raise ValueError(f"Etiqueta(s) desconhecida(s) {bad or [text]}. Válidas: {classes}")
    return [lookup[part] for part in parts]


def load_label_index() -> dict[str, tuple[str, str]]:
    """nome do ficheiro -> (split, etiqueta), a partir dos CSVs do projeto."""
    index: dict[str, tuple[str, str]] = {}
    try:
        from paths import SPLITS_DIR
    except Exception:
        return index
    for split in ("test", "val", "train"):
        csv = Path(SPLITS_DIR) / f"{split}_shot_size.csv"
        if not csv.exists():
            continue
        frame = pd.read_csv(csv)
        for path, label in zip(frame["path"], frame["label"]):
            index.setdefault(Path(str(path)).name, (split, str(label).strip()))
    return index


def load_models(run_dirs: list[Path]):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    import finetune_shotsize as fs

    device = fs.select_device()
    models, label_to_id, model_name = [], None, None
    for run in run_dirs:
        print(f"A carregar {run.name} ...")
        checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
        if label_to_id is None:
            label_to_id, model_name = checkpoint["label_to_id"], checkpoint["model_name"]
        elif checkpoint["label_to_id"] != label_to_id:
            raise SystemExit(f"{run.name} tem um mapeamento de classes diferente das outras seeds")
        backbone = AutoModel.from_pretrained(checkpoint["model_name"])
        feature_dim = getattr(backbone.config, "projection_dim", None)
        if feature_dim is None:
            feature_dim = backbone.config.vision_config.hidden_size
        model = fs.SigLIPShotSizeClassifier(backbone, feature_dim, len(label_to_id))
        model.load_state_dict(checkpoint["model_state"])
        if hasattr(model.backbone, "text_model"):  # a torre de texto não é usada: poupa memória
            del model.backbone.text_model
        models.append(model.to(device).eval())
        del checkpoint
    processor = AutoImageProcessor.from_pretrained(model_name)
    return models, processor, label_to_id, device, fs


def predict_probs(models, processor, device, fs, image_path: str) -> np.ndarray:
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    inputs = fs.move_inputs(processor(images=[image], return_tensors="pt"), device)
    with torch.no_grad():
        return np.stack([torch.softmax(m(inputs).float(), dim=1)[0].cpu().numpy() for m in models])


def report(
    image_path: str,
    probs: np.ndarray,
    classes: list[str],
    rank: dict[str, int],
    known: list[str] | None,
    known_source: str | None,
    margin: float,
    show_models: bool,
) -> None:
    mean = probs.mean(axis=0)
    order = np.argsort(-mean)
    top1, top2 = classes[order[0]], classes[order[1]]
    p1, p2 = mean[order[0]], mean[order[1]]
    votes = int((probs.argmax(axis=1) == order[0]).sum())

    print(f"\nImagem: {Path(image_path).name}")
    print(f"Previsão: {top1}  ({p1:.0%})")
    print(f"Top-2:    {top1} {p1:.0%}  |  {top2} {p2:.0%}")
    print(f"Modelos que concordam: {votes}/{len(probs)}")
    if show_models:
        votes_by_class = ", ".join(classes[i] for i in probs.argmax(axis=1))
        print(f"  por seed: {votes_by_class}")
    if abs(rank[top1] - rank[top2]) == 1 and (p1 - p2) < margin:
        print(f"Aviso: fronteira ambígua entre '{top1}' e '{top2}' (classes vizinhas, margem {p1 - p2:.0%}).")

    if known:
        label_text = " / ".join(known)
        origin = f" [{known_source}]" if known_source else ""
        if top1 in known:
            extra = " (uma das etiquetas anotadas)" if len(known) > 1 else ""
            print(f"Etiqueta conhecida: {label_text}{origin} -> COINCIDE{extra}")
        else:
            distance = min(abs(rank[top1] - rank[k]) for k in known)
            second = " A etiqueta conhecida é a 2.ª mais provável." if top2 in known else ""
            print(
                f"Etiqueta conhecida: {label_text}{origin} -> DIFERENTE "
                f"(a {distance} classe(s) de distância).{second}"
            )


def resolve_known(
    image_path: str,
    label_text: str | None,
    classes: list[str],
    index: dict[str, tuple[str, str]],
) -> tuple[list[str] | None, str | None]:
    found = index.get(Path(image_path).name)
    source = None
    if found and found[0] != "test":
        print(f"Aviso: esta imagem está no split '{found[0]}'; o modelo viu-a no treino/validação.")
    if label_text is not None:
        return parse_known(label_text, classes), None
    if found:
        return parse_known(found[1], classes), f"dos CSVs, split {found[0]}"
    return None, source


def parse_line(line: str) -> tuple[str, str | None]:
    path_part, _, label_part = line.partition("|")
    tokens = shlex.split(path_part.strip())
    if not tokens:
        raise ValueError("sem caminho de imagem")
    return tokens[0], (label_part.strip() or None)


def main() -> None:
    args = parse_args()
    run_dirs = sorted({Path(p) for pattern in args.runs for p in glob.glob(pattern) if Path(p).is_dir()})
    if not run_dirs:
        raise SystemExit(f"Nenhuma pasta de seed encontrada para {args.runs}")
    if args.single:
        run_dirs = run_dirs[:1]
    if args.label and len(args.images) != 1:
        raise SystemExit("--label só se usa com exatamente uma imagem")

    models, processor, label_to_id, device, fs = load_models(run_dirs)
    classes = [c for c, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    if set(args.class_order) != set(classes):
        raise SystemExit(f"--class-order não corresponde às classes do modelo: {classes}")
    rank = {c: i for i, c in enumerate(args.class_order)}
    index = load_label_index()
    print(f"{len(models)} modelo(s) carregado(s) em {device}.")

    def run_one(image_path: str, label_text: str | None) -> None:
        if not Path(image_path).is_file():
            print(f"Ficheiro não encontrado: {image_path}")
            return
        try:
            known, source = resolve_known(image_path, label_text, classes, index)
        except ValueError as error:
            print(error)
            return
        probs = predict_probs(models, processor, device, fs, image_path)
        report(image_path, probs, classes, rank, known, source, args.ambiguity_margin, args.show_models)

    if args.images:
        for image in args.images:
            run_one(image, args.label)
        return

    print("\nEscreve:  caminho/da/imagem.jpg | etiqueta   (a etiqueta é opcional; linha vazia ou 'q' para sair)")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line.lower() == "q":
            break
        try:
            image_path, label_text = parse_line(line)
        except ValueError as error:
            print(error)
            continue
        run_one(image_path, label_text)


if __name__ == "__main__":
    main()