from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("api", "0018_aboutstatscache"),
    ]

    operations = [
        migrations.AddField(
            model_name="job",
            name="quota_subject",
            field=models.CharField(
                blank=True,
                default="",
                help_text="Identifier used for quota accounting (IP or API-key subject).",
                max_length=128,
            ),
        ),
    ]
