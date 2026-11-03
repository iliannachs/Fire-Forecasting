"""Experiment folder handling: output location and settings snapshot."""
import json
import shutil
from pathlib import Path

from exec.cache import settings_fingerprint

# Settings that do not influence the scientific result and are therefore
# excluded from the consistency check of an existing experiment folder.
NON_SCIENTIFIC = {
    "EXPERIMENT_NAME", "PROJECT_ROOT", "OUTPUT_DIR", "CACHE_DIR",
    "EXPERIMENTS_DIR", "OVERWRITE_CACHE", "OVERWRITE_MODEL", "RUN_SHAP",
}


def collect_settings(settings):
    """All UPPERCASE settings; the SiB4 block is reduced to what is actually used."""
    info = {k: getattr(settings, k) for k in dir(settings) if k.isupper()}
    # Defaults for options that older settings files do not contain.
    info.setdefault("TUNING_SCHEME", "original")
    info.setdefault("TUNING_METRIC", "annual_r2")
    info.setdefault("TUNING_FIRST_VALIDATION_YEAR", 2010)
    info.setdefault("TUNING_BUFFER_YEARS", 1)

    sib_by_cluster = info.pop("SIB4_PREDICTORS_BY_CLUSTER")
    info["SIB4_PREDICTORS_USED"] = (
        sib_by_cluster.get(settings.CLUSTER_TO_MODEL, {}) if settings.USE_SIB4 else {}
    )
    if not settings.USE_SIB4:
        # Paths of unused data should not make two ERA5-only runs differ.
        info.pop("SIB4_DIR")
        info.pop("SIB4_FILE_PATTERN")
    return info


class Experiment:
    """
    One experiment = one named folder holding the settings snapshot, trained
    model, tables and figures of one cluster:
        outputs/experiments/<EXPERIMENT_NAME>/cluster_<N>/
    """

    def __init__(self, settings):
        self.settings = settings
        self.dir = (
            Path(settings.EXPERIMENTS_DIR)
            / settings.EXPERIMENT_NAME
            / f"cluster_{settings.CLUSTER_TO_MODEL}"
        )
        self.dir.mkdir(parents=True, exist_ok=True)

        self.info = collect_settings(settings)
        fingerprint_info = {k: v for k, v in self.info.items() if k not in NON_SCIENTIFIC}
        # Experiments created before the tuning options existed stay valid as long as
        # the original tuning is used; the extra options only count when they matter.
        if self.info["TUNING_SCHEME"] == "original":
            fingerprint_info.pop("TUNING_FIRST_VALIDATION_YEAR")
            fingerprint_info.pop("TUNING_BUFFER_YEARS")
            if self.info["TUNING_METRIC"] == "annual_r2":
                fingerprint_info.pop("TUNING_SCHEME")
                fingerprint_info.pop("TUNING_METRIC")
        else:
            if self.info["TUNING_SCHEME"] != "yearly_forward":
                fingerprint_info.pop("TUNING_FIRST_VALIDATION_YEAR")
            if self.info["TUNING_SCHEME"] != "leave_one_year_out":
                fingerprint_info.pop("TUNING_BUFFER_YEARS")
        self.fingerprint = settings_fingerprint(fingerprint_info)
        self._check_existing()
        self._write_snapshot()

    def path(self, filename):
        return str(self.dir / filename)

    def _check_existing(self):
        """Refuse to mix stored results with different settings."""
        snap = self.dir / "settings_snapshot.json"
        if not snap.exists() or self.settings.OVERWRITE_MODEL:
            return

        with open(snap) as f:
            stored = json.load(f).get("fingerprint")

        if stored != self.fingerprint:
            raise RuntimeError(
                f"Experiment folder {self.dir} was created with different settings.\n"
                "Choose a new EXPERIMENT_NAME in settings.py, or set "
                "OVERWRITE_MODEL = True to retrain and overwrite it."
            )

    def _write_snapshot(self):
        """Store the settings next to the results so every model can be traced back."""
        with open(self.dir / "settings_snapshot.json", "w") as f:
            json.dump(
                {"fingerprint": self.fingerprint, "settings": self.info},
                f, indent=2, default=str, sort_keys=True,
            )
        # Literal copy of the settings file as used for this run.
        shutil.copy(Path(self.settings.__file__), self.dir / "settings_used.py")
