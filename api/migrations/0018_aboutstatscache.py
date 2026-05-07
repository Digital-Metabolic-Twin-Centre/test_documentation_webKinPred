# Generated manually for AboutStatsCache persistent metrics cache.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0017_jobprogressstage"),
    ]

    operations = [
        migrations.CreateModel(
            name="AboutStatsCache",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("key", models.CharField(max_length=64, unique=True)),
                ("payload", models.TextField(blank=True, default="")),
                ("generated_at", models.DateTimeField(blank=True, null=True)),
                ("is_stale", models.BooleanField(default=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "About Stats Cache",
                "verbose_name_plural": "About Stats Cache",
                "ordering": ["-updated_at"],
            },
        ),
    ]
