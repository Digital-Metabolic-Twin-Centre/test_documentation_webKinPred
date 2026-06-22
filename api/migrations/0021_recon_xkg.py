"""
ReconXKG memoization feature.

Adds:
  * Job.recon_xkg audit flag (default DB);
  * ReconXkgAllowedKey allowlist table (default DB);
  * PredictionStore + SimilarityStore cache tables (prediction_store DB — see
    api/dbrouters.PredictionStoreRouter).

Apply to BOTH databases:
    python manage.py migrate
    python manage.py migrate --database=prediction_store

The router skips each CreateModel/AddField on the database it does not belong
to, so running both commands is safe and idempotent.
"""

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0020_apikey_custom_daily_limit"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="recon_xkg",
            field=models.BooleanField(
                default=False,
                help_text=(
                    "Internal: this job was served via the ReconXKG memoization "
                    "store (allowlisted keys only). Recorded for audit."
                ),
            ),
        ),
        migrations.CreateModel(
            name="ReconXkgAllowedKey",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "label",
                    models.CharField(
                        blank=True,
                        default="",
                        help_text="Optional human-readable note for this allowlist entry.",
                        max_length=100,
                    ),
                ),
                (
                    "is_active",
                    models.BooleanField(
                        default=True,
                        help_text="Disable without deleting by unchecking this.",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "api_key",
                    models.OneToOneField(
                        help_text="API key permitted to enable recon_xkg.",
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="recon_xkg_allow",
                        to="api.apikey",
                    ),
                ),
            ],
            options={
                "verbose_name": "ReconXKG Allowed Key",
                "verbose_name_plural": "ReconXKG Allowed Keys",
            },
        ),
        migrations.CreateModel(
            name="PredictionStore",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("lookup_key", models.CharField(max_length=64, unique=True)),
                ("target", models.CharField(max_length=16)),
                ("method", models.CharField(max_length=64)),
                ("model_version", models.CharField(max_length=32)),
                ("params_fingerprint", models.CharField(max_length=64)),
                ("sequence_sha256", models.CharField(max_length=64)),
                ("substrate_canon", models.TextField()),
                ("products_canon", models.TextField(blank=True, default="")),
                ("value", models.FloatField()),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "verbose_name": "Prediction Store Entry",
                "verbose_name_plural": "Prediction Store Entries",
            },
        ),
        migrations.CreateModel(
            name="SimilarityStore",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("lookup_key", models.CharField(max_length=64, unique=True)),
                ("sequence_sha256", models.CharField(db_index=True, max_length=64)),
                ("dataset_label", models.CharField(max_length=128)),
                ("mean_similarity", models.FloatField(blank=True, null=True)),
                ("max_similarity", models.FloatField(blank=True, null=True)),
                ("created_at", models.DateTimeField(default=django.utils.timezone.now)),
                ("updated_at", models.DateTimeField(default=django.utils.timezone.now)),
            ],
            options={
                "verbose_name": "Similarity Store Entry",
                "verbose_name_plural": "Similarity Store Entries",
            },
        ),
        migrations.AddIndex(
            model_name="predictionstore",
            index=models.Index(fields=["model_version"], name="predstore_modelver_idx"),
        ),
        migrations.AddIndex(
            model_name="predictionstore",
            index=models.Index(fields=["method", "target"], name="predstore_method_tgt_idx"),
        ),
    ]
