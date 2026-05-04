"""
Kill an in-flight (or stuck) job and resubmit it as a fresh Celery task.

Unlike resubmit_pending_jobs this works for any status (Pending, Processing,
Failed, …).  The DB row is deleted and recreated so that public_id and
submission_time are preserved while all runtime state is wiped clean.
JobProgressStage rows cascade-delete with the old Job row.

Usage:
    python manage.py restart_job Mbv8CWH
    python manage.py restart_job Mbv8CWH --dry-run
"""

import ast

from django.core.management.base import BaseCommand, CommandError

from api.models import Job
from api.tasks import run_multi_prediction
from api.utils.job_utils import canonicalise_targets
from webKinPred.celery import app


def _parse_task_args(raw):
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if isinstance(raw, str):
        try:
            v = ast.literal_eval(raw)
            return list(v) if isinstance(v, (list, tuple)) else []
        except Exception:
            return []
    return []


class Command(BaseCommand):
    help = "Kill a running/stuck job and resubmit it as a fresh Celery task"

    def add_arguments(self, parser):
        parser.add_argument("public_id", help="Public job ID to restart (e.g. Mbv8CWH)")
        parser.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")

    def handle(self, *args, **options):
        public_id = options["public_id"]
        dry_run = options["dry_run"]

        try:
            job = Job.objects.select_related("user").get(public_id=public_id)
        except Job.DoesNotExist:
            raise CommandError(f"No job found with public_id={public_id!r}")

        self.stdout.write(
            f"Job {job.public_id}  status={job.status}  type={job.prediction_type}  rows={job.requested_rows}"
        )

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry-run — no changes made."))
            return

        # Snapshot everything needed to recreate the row.
        snap = {
            "public_id": job.public_id,
            "prediction_type": job.prediction_type,
            "ip_address": job.ip_address,
            "requested_rows": job.requested_rows,
            "kcat_method": job.kcat_method,
            "km_method": job.km_method,
            "kcat_km_method": job.kcat_km_method,
            "handle_long_sequences": job.handle_long_sequences,
            "canonicalize_substrates": job.canonicalize_substrates,
            "user_id": job.user_id,
            "submission_time": job.submission_time,
        }

        # Revoke any active/reserved/scheduled Celery task for this job.
        self.stdout.write("Inspecting active Celery tasks…")
        try:
            insp = app.control.inspect(timeout=5)
            active = insp.active() or {}
            reserved = insp.reserved() or {}
            scheduled = insp.scheduled() or {}
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"  Inspect failed (non-fatal): {exc}"))
            active = reserved = scheduled = {}

        revoked = []
        for bucket_name, bucket in [("active", active), ("reserved", reserved)]:
            for _, tasks in bucket.items():
                for t in tasks:
                    args = _parse_task_args(t.get("args", []))
                    if args and str(args[0]) == public_id:
                        tid = t.get("id")
                        if tid:
                            app.control.revoke(tid, terminate=True, signal="SIGTERM")
                            revoked.append((bucket_name, tid))

        for _, tasks in scheduled.items():
            for t in tasks:
                req = t.get("request", {}) if isinstance(t, dict) else {}
                args = _parse_task_args(req.get("args", []))
                if args and str(args[0]) == public_id:
                    tid = req.get("id")
                    if tid:
                        app.control.revoke(tid, terminate=True, signal="SIGTERM")
                        revoked.append(("scheduled", tid))

        if revoked:
            for bucket, tid in revoked:
                self.stdout.write(f"  Revoked [{bucket}] task {tid}")
        else:
            self.stdout.write("  No active/reserved/scheduled tasks found for this job.")

        # Delete the stale row (cascades to JobProgressStage).
        self.stdout.write(f"Deleting job row for {public_id}…")
        Job.objects.filter(public_id=public_id).delete()

        # Recreate with clean runtime state, preserving identity + submission_time.
        new_job = Job(
            public_id=snap["public_id"],
            prediction_type=snap["prediction_type"],
            ip_address=snap["ip_address"],
            requested_rows=snap["requested_rows"],
            kcat_method=snap["kcat_method"],
            km_method=snap["km_method"],
            kcat_km_method=snap["kcat_km_method"],
            handle_long_sequences=snap["handle_long_sequences"],
            canonicalize_substrates=snap["canonicalize_substrates"],
            user_id=snap["user_id"],
            submission_time=snap["submission_time"],
            status="Pending",
            start_time=None,
            completion_time=None,
            error_message=None,
            total_molecules=0,
            molecules_processed=0,
            invalid_rows=0,
            total_predictions=0,
            predictions_made=0,
        )
        new_job.save()

        targets = []
        methods = {}
        if snap["kcat_method"]:
            targets.append("kcat")
            methods["kcat"] = snap["kcat_method"]
        if snap["km_method"]:
            targets.append("Km")
            methods["Km"] = snap["km_method"]
        if snap["kcat_km_method"]:
            targets.append("kcat/Km")
            methods["kcat/Km"] = snap["kcat_km_method"]

        ordered = canonicalise_targets(targets)

        run_multi_prediction.delay(
            snap["public_id"],
            ordered,
            methods,
            {},
            bool(snap["canonicalize_substrates"]),
            True,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Requeued {snap['public_id']}  targets={ordered}  methods={methods}"
            )
        )
