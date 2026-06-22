"""
Management command: recon_xkg_admin

Administer the ReconXKG memoization store: inspect stats, purge stale entries by
model version or method, and manage the API-key allowlist.

Examples
--------
    # Show counts in the store
    python manage.py recon_xkg_admin stats

    # Purge all cached predictions for a method (e.g. after a weights bump)
    python manage.py recon_xkg_admin purge --method CatPred

    # Purge a specific stale model version of a method
    python manage.py recon_xkg_admin purge --method TurNup --model-version 1

    # Purge the entire prediction store (keeps similarity cache)
    python manage.py recon_xkg_admin purge --all

    # Allowlist / de-allowlist an API key by id or key prefix
    python manage.py recon_xkg_admin allow --api-key-id 7
    python manage.py recon_xkg_admin deny --api-key-id 7
    python manage.py recon_xkg_admin list-allowed
"""

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Inspect and maintain the ReconXKG memoization store and allowlist."

    def add_arguments(self, parser):
        parser.add_argument(
            "action",
            choices=["stats", "purge", "allow", "deny", "list-allowed"],
            help="What to do.",
        )
        parser.add_argument("--method", help="Restrict to a single method key.")
        parser.add_argument(
            "--model-version",
            dest="model_version",
            help="Restrict a purge to a single model version.",
        )
        parser.add_argument(
            "--all",
            action="store_true",
            help="Purge every prediction-store entry.",
        )
        parser.add_argument(
            "--api-key-id",
            dest="api_key_id",
            type=int,
            help="ApiKey primary key for allow/deny.",
        )
        parser.add_argument(
            "--label",
            default="",
            help="Optional label for a new allowlist entry.",
        )

    def handle(self, *args, **options):
        action = options["action"]
        handler = {
            "stats": self._stats,
            "purge": self._purge,
            "allow": self._allow,
            "deny": self._deny,
            "list-allowed": self._list_allowed,
        }[action]
        handler(options)

    # ── stats ──────────────────────────────────────────────────────────────
    def _stats(self, options):
        from api.models import PredictionStore, ReconXkgAllowedKey, SimilarityStore

        self.stdout.write(f"PredictionStore entries: {PredictionStore.objects.count()}")
        self.stdout.write(f"SimilarityStore entries: {SimilarityStore.objects.count()}")
        self.stdout.write(f"Allowlisted keys:        {ReconXkgAllowedKey.objects.count()}")

        from django.db.models import Count

        by_method = (
            PredictionStore.objects.values("method", "target", "model_version")
            .annotate(n=Count("id"))
            .order_by("-n")
        )
        for row in by_method:
            self.stdout.write(
                f"  {row['method']}/{row['target']}@{row['model_version']}: {row['n']}"
            )

    # ── purge ──────────────────────────────────────────────────────────────
    def _purge(self, options):
        from api.models import PredictionStore

        qs = PredictionStore.objects.all()
        if options["all"]:
            pass
        elif options["method"] or options["model_version"]:
            if options["method"]:
                qs = qs.filter(method=options["method"])
            if options["model_version"]:
                qs = qs.filter(model_version=options["model_version"])
        else:
            raise CommandError("Specify --all, --method, and/or --model-version for purge.")

        count = qs.count()
        qs.delete()
        self.stdout.write(self.style.SUCCESS(f"Purged {count} prediction-store entr(ies)."))

    # ── allowlist ──────────────────────────────────────────────────────────
    def _allow(self, options):
        from api.models import ApiKey, ReconXkgAllowedKey

        api_key = self._require_api_key(options)
        entry, created = ReconXkgAllowedKey.objects.get_or_create(
            api_key=api_key,
            defaults={"label": options["label"], "is_active": True},
        )
        if not created:
            entry.is_active = True
            if options["label"]:
                entry.label = options["label"]
            entry.save(update_fields=["is_active", "label"])
        self.stdout.write(
            self.style.SUCCESS(f"recon_xkg enabled for ApiKey #{api_key.pk} ({api_key.key_prefix}).")
        )

    def _deny(self, options):
        from api.models import ReconXkgAllowedKey

        api_key = self._require_api_key(options)
        updated = ReconXkgAllowedKey.objects.filter(api_key=api_key).update(is_active=False)
        if updated:
            self.stdout.write(self.style.SUCCESS(f"recon_xkg disabled for ApiKey #{api_key.pk}."))
        else:
            self.stdout.write("No allowlist entry existed for that key.")

    def _list_allowed(self, options):
        from api.models import ReconXkgAllowedKey

        rows = ReconXkgAllowedKey.objects.select_related("api_key").all()
        if not rows:
            self.stdout.write("No allowlisted keys.")
            return
        for row in rows:
            state = "active" if row.is_active else "inactive"
            self.stdout.write(
                f"  ApiKey #{row.api_key_id} ({row.api_key.key_prefix}) "
                f"[{state}] {row.label}"
            )

    def _require_api_key(self, options):
        from api.models import ApiKey

        api_key_id = options.get("api_key_id")
        if not api_key_id:
            raise CommandError("--api-key-id is required for allow/deny.")
        try:
            return ApiKey.objects.get(pk=api_key_id)
        except ApiKey.DoesNotExist as exc:
            raise CommandError(f"No ApiKey with id {api_key_id}.") from exc
