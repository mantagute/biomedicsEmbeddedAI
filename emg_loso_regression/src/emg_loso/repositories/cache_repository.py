"""
repositories/cache_repository.py
--------------------------------
Camada de Armazenamento (Repository): Persistência Idempotente em `.npz` por Sujeito.

Por que persistir em `.npz`?
- Reduz o dataset de ~71 GB de CSVs brutos para ~3-5 GB binários `float32`.
- Permite que os 21 folds do LOSO carreguem todos os sujeitos em segundos,
  sem manter o SSD externo conectado durante o treino.
- Escrita atômica (`.tmp.npz` -> `os.replace`) impede arquivos corrompidos caso
  o processo seja interrompido no meio de uma extração em lote.
- Leitura com `allow_pickle=False` garante segurança e zero execução arbitrária.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from emg_loso.domain.entities import SignalProcessingConfig, SubjectDataset
from emg_loso.domain.signal.topography import extract_recording_representations
from emg_loso.repositories.csv_repository import (
    discover_subject_recordings,
    load_angles_dataframe,
    load_emg_array,
)


class NpzSubjectCacheRepository:
    """Implementação em disco do contrato `SubjectCacheRepositoryPort` usando arquivos `.npz`."""

    def __init__(self, cache_dir: str | Path, compressed: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.compressed = compressed

    def _subject_path(self, subject: str) -> Path:
        return self.cache_dir / f"{subject}.npz"

    def exists(self, subject: str) -> bool:
        """Retorna True se o arquivo `<subject>.npz` já estiver salvo em disco."""
        return self._subject_path(subject).exists()

    def save_subject(self, dataset: SubjectDataset) -> Path:
        """Persiste o `SubjectDataset` de forma atômica e retorna o caminho final."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = self._subject_path(dataset.subject)
        tmp_path = self.cache_dir / f".{dataset.subject}.tmp.npz"

        saver = np.savez_compressed if self.compressed else np.savez
        saver(
            tmp_path,
            x_flat=dataset.x_flat.astype(np.float32, copy=False),
            y_angles=dataset.y_angles.astype(np.float32, copy=False),
            rep_ids=dataset.rep_ids.astype(np.int8, copy=False),
            task_ids=dataset.task_ids.astype(np.int8, copy=False),
            freq_codes=dataset.freq_codes.astype(np.int8, copy=False),
        )
        tmp_path.replace(final_path)
        return final_path

    def load_subject(self, subject: str) -> SubjectDataset:
        """Carrega e valida o `SubjectDataset` a partir do `.npz` em disco."""
        path = self._subject_path(subject)
        if not path.exists():
            raise FileNotFoundError(f"Cache não encontrado para {subject}: {path}")

        with np.load(path, allow_pickle=False) as data:
            return SubjectDataset(
                subject=subject,
                x_flat=data["x_flat"],
                y_angles=data["y_angles"],
                rep_ids=data["rep_ids"],
                task_ids=data["task_ids"],
                freq_codes=data["freq_codes"],
            )

    def list_cached_subjects(self) -> list[str]:
        """Lista todos os sujeitos (`SubXXX`) presentes no diretório de cache."""
        if not self.cache_dir.exists():
            return []
        return sorted(p.stem for p in self.cache_dir.glob("Sub*.npz"))


def build_subject_dataset_from_csv(
    raw_data_root: str | Path,
    subject: str,
    cfg: SignalProcessingConfig = SignalProcessingConfig.hardware_causal(),
    normalize_wl: bool = True,
    extra_excluded: set[str] | frozenset[str] = frozenset(),
    verbose: bool = True,
) -> SubjectDataset:
    """Processa todas as gravações válidas de um sujeito do CSV bruto até `SubjectDataset`."""
    recordings = discover_subject_recordings(
        raw_data_root=raw_data_root,
        subject=subject,
        extra_excluded=extra_excluded,
    )
    if not recordings:
        raise FileNotFoundError(
            f"Nenhuma gravação válida encontrada para {subject} em {raw_data_root}."
        )

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    rep_list: list[np.ndarray] = []
    task_list: list[np.ndarray] = []
    freq_list: list[np.ndarray] = []

    for idx, rec in enumerate(recordings, start=1):
        emg = load_emg_array(rec.emg_path)
        angles = load_angles_dataframe(rec.angles_path)

        x_flat, _, y_angles, _ = extract_recording_representations(
            emg_array=emg,
            angles_df=angles,
            cfg=cfg,
            normalize_wl=normalize_wl,
        )

        n_win = len(x_flat)
        freq_int = 50 if rec.freq_code == "05" else 75

        x_list.append(x_flat)
        y_list.append(y_angles)
        rep_list.append(np.full(n_win, rec.rep, dtype=np.int8))
        task_list.append(np.full(n_win, rec.task_id, dtype=np.int8))
        freq_list.append(np.full(n_win, freq_int, dtype=np.int8))

        if verbose and (idx % 16 == 0 or idx == len(recordings)):
            print(f"  [{subject}] {idx}/{len(recordings)} gravações processadas...")

    return SubjectDataset(
        subject=subject,
        x_flat=np.concatenate(x_list, axis=0),
        y_angles=np.concatenate(y_list, axis=0),
        rep_ids=np.concatenate(rep_list, axis=0),
        task_ids=np.concatenate(task_list, axis=0),
        freq_codes=np.concatenate(freq_list, axis=0),
    )


def ensure_subjects_cached(
    raw_data_root: str | Path,
    cache_repo: NpzSubjectCacheRepository,
    subjects: Iterable[str],
    cfg: SignalProcessingConfig = SignalProcessingConfig.hardware_causal(),
    normalize_wl: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
) -> list[Path]:
    """Garante de forma idempotente que todos os sujeitos informados tenham cache `.npz`."""
    saved_paths: list[Path] = []
    for subject in subjects:
        if not overwrite and cache_repo.exists(subject):
            if verbose:
                print(f"[{subject}] Cache já existe — pulando.")
            saved_paths.append(cache_repo._subject_path(subject))
            continue

        if verbose:
            print(f"[{subject}] Extraindo features dos CSVs brutos...")
        ds = build_subject_dataset_from_csv(
            raw_data_root=raw_data_root,
            subject=subject,
            cfg=cfg,
            normalize_wl=normalize_wl,
            verbose=verbose,
        )
        out_path = cache_repo.save_subject(ds)
        saved_paths.append(out_path)
        if verbose:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[{subject}] Salvo em {out_path} ({ds.n_windows} janelas, {size_mb:.1f} MB)")

    return saved_paths
