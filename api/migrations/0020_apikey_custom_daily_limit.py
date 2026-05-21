from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0019_job_quota_subject"),
    ]

    operations = [
        migrations.AddField(
            model_name="apikey",
            name="custom_daily_limit",
            field=models.PositiveIntegerField(
                blank=True,
                help_text=(
                    "Optional per-key daily limit override. "
                    "Effective limit is max(user limit, key limit)."
                ),
                null=True,
            ),
        ),
    ]
