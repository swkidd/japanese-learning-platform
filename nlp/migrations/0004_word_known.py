# Generated by Django 3.0.5 on 2020-09-13 10:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nlp', '0003_word_ex_sent'),
    ]

    operations = [
        migrations.AddField(
            model_name='word',
            name='known',
            field=models.FloatField(default=0.0),
            preserve_default=False,
        ),
    ]
