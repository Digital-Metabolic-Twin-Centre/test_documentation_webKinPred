"""Allow ReconXKG to memoize deterministic row-validation failures.

Apply to both databases, as with migration 0021:

    python manage.py migrate
    python manage.py migrate --database=prediction_store

The database router applies these PredictionStore operations only to the
dedicated prediction_store database.
"""

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0021_recon_xkg"),
    ]

    operations = [
        migrations.AlterField(
            model_name="predictionstore",
            name="value",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="predictionstore",
            name="failure_reason",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddConstraint(
            model_name="predictionstore",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(value__isnull=False, failure_reason="")
                    | (models.Q(value__isnull=True) & ~models.Q(failure_reason=""))
                ),
                name="predstore_value_or_failure",
            ),
        ),
    ]
