# Generated by Django 3.0.5 on 2020-09-09 15:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('nlp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='word',
            name='hira',
            field=models.CharField(default='', max_length=30),
            preserve_default=False,
        ),
    ]
