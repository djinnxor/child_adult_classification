"""Experiment tracking for this repo: the personal mlops tracker + Weights & Biases.

mlops (~/Documents/Minje/mlops) records the lineage of the work as a tree in
$MLOPS_DOCS/<project>.json, viewable with ``mlops-web``:

    checkpoint artifact ─┐
    dataset versions ────┴─> experiment (training / finetune / evaluation) ─> best.pth.tar
    (ekyc-annotations v1 -> v2 -> ... -> ekyc-train / ekyc-valid / ekyc-test -> package)

W&B records the live curves of a training run. Both are optional:

    --no-mlops                 do not write to the mlops project
    --mlops-project NAME       project to write to (default $MLOPS_PROJECT or "mivolo")
    --mlops-parents ID,ID      extra parent nodes, e.g. the note that motivated a run
    --wandb                    log to W&B (train_mivolo.py only), see --wandb-* options

Tracking never stops the work: if mlops is not installed, or a write fails,
a warning is logged and tracking is switched off for the rest of the run.

Every mlops write is a short locked load-change-save (mlops.store.edit_project),
so a day-long training run never overwrites edits made in the dashboard
meanwhile. A run's node stores its W&B id where mlops' W&B importer looks for it,
so ``python3 -m mlops.integrations.wandb <entity/project/run_id> --project mivolo
--parents <id>`` refreshes that node instead of adding a second one; the same
command records runs trained on another machine.
"""

import hashlib
import os
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_PROJECT = os.environ.get("MLOPS_PROJECT", "mivolo")
DEFAULT_OBJECTIVE = "Fine-tune MiVOLO face-only age and gender recognition (child / adult)"
# file split names -> mlops dataset splits
SPLITS = {"train": "train", "valid": "validation", "validation": "validation", "val": "validation",
          "test": "test", "raw": "raw"}


def add_args(parser, wandb=False):
    g = parser.add_argument_group("tracking (mlops + W&B)")
    g.add_argument("--mlops-project", default=DEFAULT_PROJECT,
                   help="mlops project to record this in ($MLOPS_PROJECT)")
    g.add_argument("--mlops-root", default=None,
                   help="mlops docs folder (default: $MLOPS_DOCS, else ./mlops_docs)")
    g.add_argument("--mlops-parents", default="",
                   help="comma-separated extra parent node ids for the run this command "
                        "records, e.g. the note that motivated it")
    g.add_argument("--no-mlops", action="store_true", help="do not record in mlops")
    if wandb:
        g.add_argument("--wandb", action="store_true", help="log this run to Weights & Biases")
        g.add_argument("--wandb-project", default="mivolo")
        g.add_argument("--wandb-entity", default=None, help="user or team (default: your login)")
        g.add_argument("--wandb-name", default=None, help="run name (default: output folder name)")
        g.add_argument("--wandb-group", default=None, help="group related runs")
        g.add_argument("--wandb-tags", default="", help="comma-separated tags")
        g.add_argument("--wandb-notes", default="", help="free-text description of the run")
        g.add_argument("--wandb-mode", choices=["online", "offline"], default="online",
                       help="offline: log locally, upload later with `wandb sync`")


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _clean(metrics):
    """Drop missing values (None, NaN) so they never overwrite a real number."""
    return {k: v for k, v in metrics.items() if v is not None and v == v}


def dataset_split(split):
    return SPLITS.get(split, "other")


class Tracker:
    """One per process. ``log`` is the caller's logger, so tracking messages
    land in its console and log file."""

    def __init__(self, args, log):
        self.log = log
        self.project = args.mlops_project
        self.root = args.mlops_root
        self.extra_parents = [p.strip() for p in args.mlops_parents.split(",") if p.strip()]
        self.run_id = None          # mlops experiment node of this run
        self._t0 = time.time()
        self._prev_duration = 0.0
        self.wb = None              # wandb run
        self.mlops = False
        if not args.no_mlops:
            try:
                import mlops  # noqa: F401
                self.mlops = True
            except ImportError:
                log.info("mlops not installed: lineage tracking off "
                         "(pip install -e ~/Documents/Minje/mlops, or pass --no-mlops)")

    # ------------------------------------------------------------------ mlops
    @contextmanager
    def _project(self):
        from mlops.store import create_project, edit_project, get_project, project_slug

        slug = project_slug(self.project)
        if get_project(slug, self.root) is None:
            create_project(self.project, DEFAULT_OBJECTIVE, self.root)
            self.log.info(f"mlops: created project '{self.project}'")
        with edit_project(slug, self.root) as writer:
            yield writer

    def _do(self, what, fn):
        """Run fn(writer) inside a locked edit; on any error warn and stop tracking."""
        if not self.mlops:
            return None
        try:
            with self._project() as writer:
                return fn(writer)
        except Exception as e:  # tracking must never break training
            self.log.info(f"WARNING: mlops {what} failed ({type(e).__name__}: {e}); "
                          f"mlops tracking is off for the rest of this run")
            self.mlops = False
            return None

    def _existing(self, writer, ids, extra=False):
        """The given node ids (plus --mlops-parents if ``extra``) that exist, or
        None, which mlops resolves to the natural parent: the previous dataset
        version, the run, the datasets, or the project root."""
        from mlops.nodes import find

        wanted = [p for p in [*(ids or []), *(self.extra_parents if extra else [])] if p]
        found = [p for i, p in enumerate(wanted) if p not in wanted[:i] and find(writer.nodes, p)]
        for p in set(wanted) - set(found):
            self.log.info(f"WARNING: mlops parent '{p}' not found in '{self.project}', ignored")
        return found or None

    def dataset(self, name, ref, split, n_items=None, note="", parents=None, key=""):
        """Node id of the dataset version for this file content, adding a new
        version when the file changed. ``key`` distinguishes subsets of the
        same file (e.g. --limit)."""
        ref = str(Path(ref).expanduser().resolve())
        digest = file_sha256(ref)
        content = f"{digest}:{key}" if key else digest

        def fn(w):
            for node in w.dataset_versions:
                d = node["data"]
                if d.get("name") == name and d.get("data_ref") == ref and d.get("content") == content:
                    return node["id"]
            summary = "; ".join(s for s in (
                f"{n_items} images" if n_items is not None else "", note,
                f"sha256 {digest[:12]}") if s)
            node = w.add_dataset(name, ref, split=split, note=summary,
                                 parents=self._existing(w, parents))
            w.edit_node(node, content=content, sha256=digest, n_items=n_items)
            self.log.info(f"mlops: dataset {name} v{node['data']['version']} -> {node['id']}")
            return node["id"]

        return self._do(f"dataset {name}", fn)

    def latest(self, name):
        return self._do("lookup", lambda w: (w.latest_dataset(name) or {}).get("id"))

    def checkpoint(self, path):
        """Artifact node of a model file; registers files it has not seen."""
        ref = str(Path(path).expanduser().resolve())

        def fn(w):
            for node in w.artifact_nodes:
                if node["data"].get("artifact_ref") == ref:
                    return node["id"]
            node = w.add_artifact(Path(ref).name, ref, kind="model",
                                  note="checkpoint first used here (not produced by a tracked run)")
            self.log.info(f"mlops: registered checkpoint {Path(ref).name} -> {node['id']}")
            return node["id"]

        return self._do("checkpoint", fn)

    def start_run(self, title, model_name, mode, parameters, datasets=None, parents=None,
                  architecture="", notes=""):
        """New experiment node (status running); ``datasets`` maps node id -> role."""
        def fn(w):
            node = w.add_experiment(
                model_name, parents=self._existing(w, parents, extra=True), mode=mode,
                datasets=datasets, parameters=parameters, architecture=architecture,
                notes=notes, title=title, status="running")
            return node["id"]

        self.run_id = self._do("start run", fn)
        self._t0 = time.time()
        if self.run_id:
            self.log.info(f"mlops: run {self.run_id} '{title}' in project '{self.project}'")
        return self.run_id

    def resume_run(self, run_id):
        def fn(w):
            node = w.get_node(run_id)
            self._prev_duration = float(node["data"].get("duration_seconds") or 0)
            w.edit_node(node, status="running", error="")
            return node["id"]

        self.run_id = self._do("resume run", fn)
        self._t0 = time.time()
        if self.run_id:
            self.log.info(f"mlops: continuing run {self.run_id}")
        return self.run_id

    def log_run(self, metrics=None, parameters=None, **data):
        """Merge metrics / parameters into the run; other fields are replaced."""
        if not self.run_id:
            return

        def fn(w):
            run = w.log_into(self.run_id)
            if metrics:
                run.log_metrics(_clean(metrics))  # drop NaN
            if parameters:
                run.log_params(parameters)
            w.edit_node(self.run_id, duration_seconds=round(
                self._prev_duration + time.time() - self._t0, 1), **data)

        self._do("log", fn)

    def finish_run(self, status, error="", metrics=None):
        self.log_run(metrics=metrics, status=status, error=error)
        if self.run_id:
            self.log.info(f"mlops: run {self.run_id} {status}")

    def artifact(self, name, ref, kind="model", note="", parents=None, from_run=True):
        def fn(w):
            node = w.add_artifact(
                name, str(Path(ref).expanduser().resolve()), kind=kind, note=note,
                run_id=self.run_id if from_run else None, parents=self._existing(w, parents))
            return node["id"]

        node_id = self._do(f"artifact {name}", fn)
        if node_id:
            self.log.info(f"mlops: artifact {name} -> {node_id}")
        return node_id

    def experiment(self, title, model_name, mode, parameters, metrics, datasets=None,
                   parents=None, notes=""):
        """A finished experiment in one go (e.g. a prediction pass)."""
        def fn(w):
            node = w.add_experiment(
                model_name, parents=self._existing(w, parents, extra=True), mode=mode,
                datasets=datasets,
                metrics=_clean(metrics), parameters=parameters,
                notes=notes, title=title)
            w.edit_node(node, duration_seconds=round(time.time() - self._t0, 1))
            return node["id"]

        node_id = self._do("experiment", fn)
        if node_id:
            self.log.info(f"mlops: experiment '{title}' -> {node_id}")
        return node_id

    def remove(self, node_id):
        self._do("remove", lambda w: w.delete_node(node_id))

    def attach_path(self, node_id, path, label=""):
        if node_id:
            self._do("attach", lambda w: w.attach_path(node_id, str(Path(path).resolve()), label))

    def _attach_link_once(self, node_id, url, label):
        def fn(w):
            if not any(a.get("href") == url for a in w.get_node(node_id).get("attachments", [])):
                w.attach_link(node_id, url, label)
        self._do("attach", fn)

    # ------------------------------------------------------------------ W&B
    def init_wandb(self, args, config, job_type, run_dir, resume_id=None, name=None):
        if not getattr(args, "wandb", False):
            return None
        try:
            import wandb
        except ImportError:
            self.log.info("WARNING: --wandb given but wandb is not installed (pip install wandb); "
                          "W&B logging off")
            return None
        try:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
            self.wb = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.wandb_name or name or Path(args.output).name, group=args.wandb_group,
                tags=[t.strip() for t in args.wandb_tags.split(",") if t.strip()] or None,
                notes=args.wandb_notes or None, mode=args.wandb_mode, job_type=job_type,
                config=config, dir=str(run_dir),
                id=resume_id, resume="allow" if resume_id else None)
        except Exception as e:
            self.log.info(f"WARNING: wandb.init failed ({type(e).__name__}: {e}); W&B logging off")
            self.wb = None
            return None
        # step-level curves against the optimiser step, epoch-level against the
        # epoch, so a resumed run never logs a step "back in time"
        self.wb.define_metric("global_step")
        self.wb.define_metric("epoch")
        self.wb.define_metric("train/*", step_metric="global_step")
        for prefix in ("val/*", "epoch/*"):
            self.wb.define_metric(prefix, step_metric="epoch")
        self.wb.define_metric("val/mae", summary="min")
        self.log.info(f"W&B: {'resumed' if resume_id else 'logging to'} "
                      f"{self.wb.url or f'{args.wandb_project} (offline, upload with wandb sync)'}")
        if self.run_id:
            self.log_run(wandb=self.wandb_info)
            if self.wb.url:
                self._attach_link_once(self.run_id, self.wb.url, f"W&B: {self.wb.name}")
        return self.wb.id

    @property
    def wandb_info(self):
        if self.wb is None:
            return {}
        return {"id": self.wb.id, "name": self.wb.name, "entity": self.wb.entity,
                "project": self.wb.project, "url": self.wb.url or "", "state": "running"}

    def wb_log(self, data):
        if self.wb is not None:
            try:
                self.wb.log(data)
            except Exception as e:
                self.log.info(f"WARNING: W&B log failed ({e}); W&B logging off")
                self.wb = None

    def wb_summary(self, data):
        if self.wb is not None:
            self.wb.summary.update(data)

    def finish_wandb(self, exit_code=0):
        if self.wb is None:
            return
        info = {**self.wandb_info, "state": "finished" if exit_code == 0 else "failed"}
        try:
            self.wb.finish(exit_code=exit_code)
        except Exception as e:
            self.log.info(f"WARNING: wandb.finish failed: {e}")
        self.wb = None
        if self.run_id:
            self.log_run(wandb=info)
